import torch, cv2, numpy as np, scipy.special, os, sys
from PIL import Image
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.model import parsingNet
from data.constant import culane_row_anchor

_model = None
_device = None

def _get_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg_path = os.path.join(PROJECT_ROOT, 'configs', 'culane.py')
        from utils.config import Config
        cfg = Config.fromfile(cfg_path)
        model_path = os.path.join(PROJECT_ROOT, 'culane_18.pth')

        cls_num = 18
        _model = parsingNet(
            pretrained=False, backbone='18',
            cls_dim=(cfg.griding_num + 1, cls_num, cfg.num_lanes),
            use_aux=False,
            size=(getattr(cfg, 'img_h', 288), getattr(cfg, 'img_w', 800))
        ).to(_device)

        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location='cpu')['model']
            comp = {}
            for k, v in sd.items():
                comp[k[7:] if k.startswith('module.') else k] = v

            cls_out_dim = 0
            for k, v in comp.items():
                if 'cls' in k and 'weight' in k and v.ndim == 2 and v.shape[0] > cls_out_dim:
                    cls_out_dim = v.shape[0]
            if cls_out_dim > 1000:
                auto_griding = cls_out_dim // (cls_num * 4) - 1
                _model = parsingNet(
                    pretrained=False, backbone='18',
                    cls_dim=(auto_griding + 1, cls_num, 4), use_aux=False
                ).to(_device)
            _model.load_state_dict(comp, strict=False)
        _model.eval()
    return _model, _device

_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def _extract_lane_points(img):
    """Run model inference, return raw lane point positions (no drawing)"""
    model, device = _get_model()
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = _transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    griding_num = out.shape[1] - 1
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]

    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    row_anchor = culane_row_anchor
    cls_num = len(row_anchor)
    col_sample_w = 800 / griding_num

    detected_lanes = 0
    all_lane_points = []
    for lane_idx in range(min(out_j.shape[1], 4)):
        mask = out_j[:, lane_idx] != 0
        if np.sum(mask) > 2:
            pts_x = []
            pts_y = []
            for k in range(out_j.shape[0]):
                if out_j[k, lane_idx] > 0:
                    px = int(out_j[k, lane_idx] * col_sample_w * w / 800) - 1
                    py = int(h * (row_anchor[cls_num - 1 - min(k, cls_num - 1)] / 288)) - 1
                    if 0 <= px < w and 0 <= py < h:
                        pts_x.append(px)
                        pts_y.append(py)
            if len(pts_x) > 3:
                all_lane_points.append((np.array(pts_x), np.array(pts_y)))
                detected_lanes += 1

    dl_left_curve = None
    dl_right_curve = None
    if len(all_lane_points) >= 2:
        pts0 = all_lane_points[0]
        pts1 = all_lane_points[1]
        if np.mean(pts0[0]) < np.mean(pts1[0]):
            left_pts, right_pts = pts0, pts1
        else:
            left_pts, right_pts = pts1, pts0
        try:
            lf = np.polyfit(left_pts[1], left_pts[0], 2)
            dl_left_curve = round(float(lf[0]), 8)
        except:
            pass
        try:
            rf = np.polyfit(right_pts[1], right_pts[0], 2)
            dl_right_curve = round(float(rf[0]), 8)
        except:
            pass
    elif len(all_lane_points) == 1:
        try:
            f = np.polyfit(all_lane_points[0][1], all_lane_points[0][0], 2)
            dl_left_curve = round(float(f[0]), 8)
        except:
            pass

    return all_lane_points, detected_lanes, dl_left_curve, dl_right_curve


def detect_dl(img):
    result = img.copy()
    all_lane_points, detected_lanes, dl_left_curve, dl_right_curve = _extract_lane_points(img)

    for pts_x, pts_y in all_lane_points:
        for i in range(len(pts_x)):
            cv2.circle(result, (pts_x[i], pts_y[i]), 5, (0, 255, 0), -1)

    cv2.putText(result, f'DL LANES: {detected_lanes}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if dl_left_curve is not None:
        cv2.putText(result, f'Left curve: {dl_left_curve}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if dl_right_curve is not None:
        cv2.putText(result, f'Right curve: {dl_right_curve}', (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return result, detected_lanes > 0, (None, None, None), dl_left_curve, dl_right_curve, detected_lanes


class LaneDLDetector:
    """DL 视频检测器：绿点 EMA 平滑 → 连续折线连接"""

    def __init__(self, smooth_window=8):
        self.max_hold = 15
        self.ema_points = {}
        self.detected = False
        self._miss_count = 0

    @staticmethod
    def _sort_lanes(all_points):
        return sorted(all_points, key=lambda p: np.mean(p[0]))

    def _smooth_raw_points(self, sorted_pts):
        result = []
        for i, (nx, ny) in enumerate(sorted_pts):
            if len(nx) < 3:
                continue
            key = f'l{i}'
            prev = self.ema_points.get(key)
            if prev is None:
                self.ema_points[key] = (nx.astype(float).copy(), ny.astype(float).copy())
                result.append((nx, ny))
            else:
                px_arr, py_arr = prev
                ex = px_arr.copy()
                ey = py_arr.copy()
                for ki in range(len(ny)):
                    matched = False
                    for kj in range(len(py_arr)):
                        if abs(py_arr[kj] - ny[ki]) < 12:
                            ex[kj] = ex[kj] * 0.4 + nx[ki] * 0.6
                            matched = True
                            break
                    if not matched:
                        ex = np.append(ex, float(nx[ki]))
                        ey = np.append(ey, float(ny[ki]))
                sort_idx = np.argsort(ey)
                ex = ex[sort_idx]
                ey = ey[sort_idx]
                self.ema_points[key] = (ex, ey)
                result.append((ex.astype(int), ey.astype(int)))
        return result

    def detect(self, img):
        h, w = img.shape[:2]
        all_points, detected_lanes, dl_left, dl_right = _extract_lane_points(img)
        result = img.copy()

        has_new = len(all_points) > 0
        if has_new:
            self._miss_count = 0
            sorted_pts = self._sort_lanes(all_points)
            smoothed = self._smooth_raw_points(sorted_pts)
            self.detected = len(smoothed) > 0
        elif self.detected and self._miss_count < self.max_hold:
            self._miss_count += 1
            smoothed = []
            for key in sorted(self.ema_points.keys()):
                ex, ey = self.ema_points[key]
                smoothed.append((ex.astype(int), ey.astype(int)))
        else:
            self.detected = False
            self._miss_count = 0
            self.ema_points.clear()
            smoothed = []

        for pts_x, pts_y in smoothed:
            for i in range(len(pts_x)):
                cv2.circle(result, (int(pts_x[i]), int(pts_y[i])), 5, (0, 255, 0), -1)
            pts_arr = np.int_([np.transpose(np.vstack([pts_x, pts_y]))])
            cv2.polylines(result, pts_arr, False, (0, 255, 0), 3)

        status_color = (0, 255, 255) if self._miss_count > 0 else ((0, 255, 0) if self.detected else (0, 0, 255))
        status = f'DL: {"OK" if self.detected else "LOST"}'
        if self._miss_count > 0:
            status += f' (hold {self._miss_count})'
        cv2.putText(result, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        if dl_left is not None:
            cv2.putText(result, f'Left curve: {dl_left}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if dl_right is not None:
            cv2.putText(result, f'Right curve: {dl_right}', (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        curves = (None, None, None)
        return result, self.detected, curves
