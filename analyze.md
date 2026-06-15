import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class EdgeSpreadWidthDetector {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static class Config {
        // contour 위에서 몇 픽셀 간격으로 샘플링할지
        public int contourSampleStep = 5;

        // tangent 계산 시 앞뒤 몇 포인트를 볼지
        public int tangentGap = 3;

        // 경계선을 기준으로 양쪽 몇 픽셀까지 profile을 볼지
        public double normalRadius = 12.0;

        // profile 샘플링 간격. 0.5면 sub-pixel 샘플링
        public double normalSampleStep = 0.5;

        // 양쪽 plateau 평균 계산에 사용할 샘플 개수
        public int endpointAvgCount = 5;

        // 경계 양쪽 밝기 차이가 이 값보다 작으면 유효하지 않은 edge로 제외
        public double minContrast = 15.0;

        // 너무 작은 contour 제거
        public double minContourArea = 50.0;

        // mask 자동 반전 여부
        public boolean autoInvertMask = true;

        // ESW가 이 값보다 크면 blur로 판정
        public double blurThresholdPx = 4.0;
    }

    public static class Result {
        public double meanWidth;
        public double medianWidth;
        public double stdWidth;
        public int sampleCount;
        public boolean blurry;
        public List<Double> widths;

        @Override
        public String toString() {
            return "Result{" +
                    "meanWidth=" + meanWidth +
                    ", medianWidth=" + medianWidth +
                    ", stdWidth=" + stdWidth +
                    ", sampleCount=" + sampleCount +
                    ", blurry=" + blurry +
                    '}';
        }
    }

    public static Result evaluateImage(String imagePath, Config config) {
        Mat src = Imgcodecs.imread(imagePath);

        if (src.empty()) {
            throw new IllegalArgumentException("이미지를 읽을 수 없습니다: " + imagePath);
        }

        Mat gray = new Mat();

        if (src.channels() == 3) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = src.clone();
        }

        Mat mask = createObjectMaskByOtsu(gray, config);

        return evaluate(gray, mask, config);
    }

    public static Result evaluate(Mat gray, Mat mask, Config config) {
        if (gray.empty()) {
            throw new IllegalArgumentException("gray image is empty");
        }

        if (mask.empty()) {
            throw new IllegalArgumentException("mask is empty");
        }

        if (gray.channels() != 1) {
            throw new IllegalArgumentException("gray image must be single channel");
        }

        if (mask.channels() != 1) {
            throw new IllegalArgumentException("mask must be single channel");
        }

        Mat contourMask = mask.clone();

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(
                contourMask,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_NONE
        );

        List<Double> widths = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);

            if (area < config.minContourArea) {
                continue;
            }

            List<Point> points = contour.toList();

            if (points.size() < config.tangentGap * 2 + 1) {
                continue;
            }

            for (int i = 0; i < points.size(); i += config.contourSampleStep) {
                Double width = measureWidthAtContourPoint(
                        gray,
                        points,
                        i,
                        config
                );

                if (width != null && !width.isNaN() && !width.isInfinite()) {
                    widths.add(width);
                }
            }
        }

        return summarize(widths, config.blurThresholdPx);
    }

    private static Mat createObjectMaskByOtsu(Mat gray, Config config) {
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(3, 3), 0);

        Mat mask = new Mat();

        Imgproc.threshold(
                blurred,
                mask,
                0,
                255,
                Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU
        );

        if (config.autoInvertMask) {
            int total = mask.rows() * mask.cols();
            int white = Core.countNonZero(mask);
            double whiteRatio = white / (double) total;

            /*
             * 배경이 흰색으로 잡힌 경우가 많으면 반전.
             * 물체가 이미지 대부분을 차지하는 경우에는 이 조건을 조정해야 함.
             */
            if (whiteRatio > 0.6) {
                Core.bitwise_not(mask, mask);
            }
        }

        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                new Size(3, 3)
        );

        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

        return mask;
    }

    private static Double measureWidthAtContourPoint(
            Mat gray,
            List<Point> contourPoints,
            int index,
            Config config
    ) {
        int n = contourPoints.size();

        int prevIndex = mod(index - config.tangentGap, n);
        int nextIndex = mod(index + config.tangentGap, n);

        Point p = contourPoints.get(index);
        Point prev = contourPoints.get(prevIndex);
        Point next = contourPoints.get(nextIndex);

        double tx = next.x - prev.x;
        double ty = next.y - prev.y;

        double tangentLength = Math.sqrt(tx * tx + ty * ty);

        if (tangentLength < 1e-6) {
            return null;
        }

        tx /= tangentLength;
        ty /= tangentLength;

        // tangent에 수직인 normal vector
        double nx = -ty;
        double ny = tx;

        List<Double> positions = new ArrayList<>();
        List<Double> intensities = new ArrayList<>();

        for (double d = -config.normalRadius;
             d <= config.normalRadius + 1e-9;
             d += config.normalSampleStep) {

            double x = p.x + nx * d;
            double y = p.y + ny * d;

            if (!isInside(gray, x, y)) {
                return null;
            }

            double value = getBilinearGray(gray, x, y);

            positions.add(d);
            intensities.add(value);
        }

        if (intensities.size() < config.endpointAvgCount * 2 + 2) {
            return null;
        }

        double[] pos = toArray(positions);
        double[] profile = toArray(intensities);

        profile = smoothMovingAverage(profile, 3);

        return calcTenToNinetyWidth(
                pos,
                profile,
                config.endpointAvgCount,
                config.minContrast
        );
    }

    private static Double calcTenToNinetyWidth(
            double[] pos,
            double[] profile,
            int endpointAvgCount,
            double minContrast
    ) {
        double startMean = mean(profile, 0, endpointAvgCount);
        double endMean = mean(profile, profile.length - endpointAvgCount, profile.length);

        double delta = endMean - startMean;

        if (Math.abs(delta) < minContrast) {
            return null;
        }

        double level10 = startMean + delta * 0.10;
        double level90 = startMean + delta * 0.90;

        boolean increasing = delta > 0;

        Double x10 = findCrossing(pos, profile, level10, increasing);
        Double x90 = findCrossing(pos, profile, level90, increasing);

        if (x10 == null || x90 == null) {
            return null;
        }

        double width = Math.abs(x90 - x10);

        if (width <= 0) {
            return null;
        }

        return width;
    }

    private static Double findCrossing(
            double[] pos,
            double[] profile,
            double level,
            boolean increasing
    ) {
        for (int i = 1; i < profile.length; i++) {
            double y0 = profile[i - 1];
            double y1 = profile[i];

            boolean crossed;

            if (increasing) {
                crossed = y0 <= level && y1 >= level;
            } else {
                crossed = y0 >= level && y1 <= level;
            }

            if (crossed) {
                double x0 = pos[i - 1];
                double x1 = pos[i];

                if (Math.abs(y1 - y0) < 1e-9) {
                    return x0;
                }

                double ratio = (level - y0) / (y1 - y0);

                return x0 + ratio * (x1 - x0);
            }
        }

        return null;
    }

    private static Result summarize(List<Double> widths, double blurThresholdPx) {
        Result result = new Result();
        result.widths = widths;
        result.sampleCount = widths.size();

        if (widths.isEmpty()) {
            result.meanWidth = 0.0;
            result.medianWidth = 0.0;
            result.stdWidth = 0.0;
            result.blurry = true;
            return result;
        }

        Collections.sort(widths);

        double sum = 0.0;
        for (double w : widths) {
            sum += w;
        }

        result.meanWidth = sum / widths.size();

        if (widths.size() % 2 == 1) {
            result.medianWidth = widths.get(widths.size() / 2);
        } else {
            int mid = widths.size() / 2;
            result.medianWidth = (widths.get(mid - 1) + widths.get(mid)) / 2.0;
        }

        double variance = 0.0;
        for (double w : widths) {
            double diff = w - result.meanWidth;
            variance += diff * diff;
        }

        variance /= widths.size();
        result.stdWidth = Math.sqrt(variance);

        /*
         * 평균보다 median을 기준으로 판정하는 것을 추천.
         * 일부 이상 edge point가 있어도 median은 덜 흔들림.
         */
        result.blurry = result.medianWidth >= blurThresholdPx;

        return result;
    }

    private static boolean isInside(Mat mat, double x, double y) {
        return x >= 0 &&
                y >= 0 &&
                x < mat.cols() - 1 &&
                y < mat.rows() - 1;
    }

    private static double getBilinearGray(Mat gray, double x, double y) {
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);

        int x1 = x0 + 1;
        int y1 = y0 + 1;

        double dx = x - x0;
        double dy = y - y0;

        double v00 = gray.get(y0, x0)[0];
        double v10 = gray.get(y0, x1)[0];
        double v01 = gray.get(y1, x0)[0];
        double v11 = gray.get(y1, x1)[0];

        double v0 = v00 * (1.0 - dx) + v10 * dx;
        double v1 = v01 * (1.0 - dx) + v11 * dx;

        return v0 * (1.0 - dy) + v1 * dy;
    }

    private static double[] smoothMovingAverage(double[] values, int radius) {
        double[] result = new double[values.length];

        for (int i = 0; i < values.length; i++) {
            int from = Math.max(0, i - radius);
            int to = Math.min(values.length - 1, i + radius);

            double sum = 0.0;
            int count = 0;

            for (int j = from; j <= to; j++) {
                sum += values[j];
                count++;
            }

            result[i] = sum / count;
        }

        return result;
    }

    private static double mean(double[] values, int from, int to) {
        double sum = 0.0;
        int count = 0;

        for (int i = from; i < to; i++) {
            sum += values[i];
            count++;
        }

        return count == 0 ? 0.0 : sum / count;
    }

    private static double[] toArray(List<Double> list) {
        double[] arr = new double[list.size()];

        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }

        return arr;
    }

    private static int mod(int value, int size) {
        int m = value % size;
        return m < 0 ? m + size : m;
    }

    public static void main(String[] args) {
        Config config = new Config();

        /*
         * 예시 기준값.
         * median ESW가 4px 이상이면 흐림으로 판정.
         * 실제로는 양품/불량 이미지로 threshold를 다시 잡는 것을 추천.
         */
        config.blurThresholdPx = 4.0;

        Result result = evaluateImage("sample.png", config);

        System.out.println(result);

        if (result.blurry) {
            System.out.println("판정: 경계선 흐림 / defocus 가능성 높음");
        } else {
            System.out.println("판정: 경계선 선명");
        }
    }
}