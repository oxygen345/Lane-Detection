import cv2
import numpy as np
from collections import deque

class LaneCVDetector:
    """
    传统计算机视觉车道线检测：
    透视变换 → 阈值提取 → 滑动窗口搜索 → 多项式曲线拟合 → 滑动平均平滑 → 可视化叠加

    视频稳定策略：
    1. 拟合合法性校验 —— 弯率/宽度异常值丢弃
    2. 针对性搜索 —— 有历史时跳过直方图，从上次位置附近搜索
    3. 单侧丢失回退 —— 一侧找不到用历史平滑值补上
    4. 完全丢失回退 —— 检测失败复用上次成功结果（最多3帧）
    """

    def __init__(self, smooth_window=5):
        self.left_fit = None
        self.right_fit = None
        self.recent_left_fits = deque(maxlen=smooth_window)
        self.recent_right_fits = deque(maxlen=smooth_window)
        self.detected = False
        self._miss_count = 0

    def _threshold(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = np.absolute(sobel_x)
        scaled_sobel = np.uint8(255 * abs_sobel_x / (np.max(abs_sobel_x) + 1e-6))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 90) & (s_channel <= 255)] = 1

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= 80) & (l_channel <= 255)] = 1

        combined = np.zeros_like(grad_binary)
        combined[((grad_binary == 1) & (l_binary == 1)) | (s_binary == 1)] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return combined

    def _perspective_transform(self, img, src=None, dst=None):
        h, w = img.shape[:2]
        if src is None:
            src = np.float32([
                [int(w * 0.12), int(h * 0.95)],
                [int(w * 0.43), int(h * 0.62)],
                [int(w * 0.57), int(h * 0.62)],
                [int(w * 0.93), int(h * 0.95)],
            ])
        if dst is None:
            dst = np.float32([
                [int(w * 0.20), h],
                [int(w * 0.20), 0],
                [int(w * 0.80), 0],
                [int(w * 0.80), h],
            ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.src_pts = src
        self.dst_pts = dst
        warped = cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def _find_lane_base(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def _find_lane_base_targeted(self, binary_warped, left_fit, right_fit):
        """Use previous fit to seed search positions (stable between frames)"""
        h = binary_warped.shape[0]
        ploty = np.linspace(0, h - 1, h)
        left_base = int(left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2])
        right_base = int(right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2])
        left_base = max(0, min(left_base, binary_warped.shape[1] - 1))
        right_base = max(0, min(right_base, binary_warped.shape[1] - 1))
        return left_base, right_base

    def _validate_fit(self, fit, lane_base, img_w, side='left'):
        """Reject obviously bad fits"""
        if fit is None:
            return False
        a, b, c = fit
        # curvature should not be insane
        if abs(a) > 0.01:
            return False
        # lane at bottom should be within reasonable range of base
        h_check = 590
        x_bottom = a * h_check ** 2 + b * h_check + c
        margin = img_w * 0.40
        if abs(x_bottom - lane_base) > margin:
            return False
        # lane should start from a reasonable x position
        if side == 'left' and (x_bottom < -50 or x_bottom > img_w * 0.6):
            return False
        if side == 'right' and (x_bottom < img_w * 0.4 or x_bottom > img_w + 50):
            return False
        return True

    def _sliding_window(self, binary_warped, leftx_base, rightx_base):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        h, w = binary_warped.shape
        nwindows = 9
        margin = 80
        minpix = 20
        window_height = h // nwindows

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

        return left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img

    def _poly_fit(self, left_inds, right_inds, nonzerox, nonzeroy, h):
        left_fit = None
        right_fit = None
        ploty = np.linspace(0, h - 1, h)

        if len(left_inds) > 30:
            leftx = nonzerox[left_inds]
            lefty = nonzeroy[left_inds]
            left_fit = np.polyfit(lefty, leftx, 2)

        if len(right_inds) > 30:
            rightx = nonzerox[right_inds]
            righty = nonzeroy[right_inds]
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, ploty

    def _smooth(self, left_fit, right_fit):
        """Smooth fits and store validated ones"""
        smoothed_left = left_fit
        smoothed_right = right_fit

        if left_fit is not None:
            self.recent_left_fits.append(left_fit)
        if right_fit is not None:
            self.recent_right_fits.append(right_fit)

        if left_fit is None and len(self.recent_left_fits) > 0:
            smoothed_left = np.mean(self.recent_left_fits, axis=0)
        elif len(self.recent_left_fits) > 1:
            smoothed_left = np.mean(self.recent_left_fits, axis=0)

        if right_fit is None and len(self.recent_right_fits) > 0:
            smoothed_right = np.mean(self.recent_right_fits, axis=0)
        elif len(self.recent_right_fits) > 1:
            smoothed_right = np.mean(self.recent_right_fits, axis=0)

        return smoothed_left, smoothed_right

    def detect(self, img, draw_windows=False):
        h, w = img.shape[:2]
        binary = self._threshold(img)
        binary_warped = self._perspective_transform(binary)

        bw_h, bw_w = binary_warped.shape

        # Determine search strategy
        have_history = (len(self.recent_left_fits) > 0 and len(self.recent_right_fits) > 0)
        if have_history:
            smoothed_l = np.mean(self.recent_left_fits, axis=0)
            smoothed_r = np.mean(self.recent_right_fits, axis=0)
            leftx_base, rightx_base = self._find_lane_base_targeted(
                binary_warped, smoothed_l, smoothed_r)
        else:
            leftx_base, rightx_base = self._find_lane_base(binary_warped)

        left_inds, right_inds, nx, ny, window_img = self._sliding_window(
            binary_warped, leftx_base, rightx_base)

        min_points = 30
        left_fit_raw, right_fit_raw, ploty = self._poly_fit(left_inds, right_inds, nx, ny, h)

        # Validate fits
        left_ok = self._validate_fit(left_fit_raw, leftx_base, bw_w, 'left')
        right_ok = self._validate_fit(right_fit_raw, rightx_base, bw_w, 'right')

        left_fit = left_fit_raw if left_ok else None
        right_fit = right_fit_raw if right_ok else None

        # Smooth and handle missing sides
        left_fit, right_fit = self._smooth(left_fit, right_fit)

        result = None
        left_fitx = None
        right_fitx = None

        if left_fit is not None and right_fit is not None:
            self.detected = True
            self._miss_count = 0
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h))
            result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

            if draw_windows:
                unwarped_window = cv2.warpPerspective(window_img, self.Minv, (w, h))
                result = cv2.addWeighted(result, 1, unwarped_window, 0.3, 0)
        elif self.detected and self._miss_count < 3:
            # Fallback: use last successful smoothed fit
            self._miss_count += 1
            if len(self.recent_left_fits) > 0:
                left_fit = np.mean(self.recent_left_fits, axis=0)
                left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            if len(self.recent_right_fits) > 0:
                right_fit = np.mean(self.recent_right_fits, axis=0)
                right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            if left_fitx is not None and right_fitx is not None:
                warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
                color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
                pts = np.hstack((pts_left, pts_right))
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 165, 255))
                newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h))
                result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
            else:
                result = img.copy()
        else:
            self.detected = False
            self._miss_count = 0
            self.recent_left_fits.clear()
            self.recent_right_fits.clear()
            result = img.copy()

        status_color = (0, 255, 255) if (self._miss_count > 0 and self.detected) else \
                       ((0, 255, 0) if self.detected else (0, 0, 255))
        status_text = f'LANES: {"OK" if self.detected else "LOST"}'
        if self._miss_count > 0:
            status_text += f' (hold {self._miss_count})'
        cv2.putText(result, status_text,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        if left_fit is not None and self.detected:
            cv2.putText(result, f'L-curve: {left_fit[0]:.6f}',
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if right_fit is not None and self.detected:
            cv2.putText(result, f'R-curve: {right_fit[0]:.6f}',
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return result, self.detected, (left_fitx, right_fitx, ploty)
