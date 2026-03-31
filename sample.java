import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class FfmpegFirstFrameExtractor {

    public static Path extractFirstFrame(Path ffmpegExe, Path inputVideo, Path outputImage)
            throws IOException, InterruptedException {

        Files.createDirectories(outputImage.getParent());

        ProcessBuilder pb = new ProcessBuilder(
                ffmpegExe.toString(),
                "-y",
                "-i", inputVideo.toString(),
                "-frames:v", "1",
                outputImage.toString()
        );

        // stderr 를 stdout 으로 합침
        pb.redirectErrorStream(true);

        Process process = pb.start();

        StringBuilder log = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), Charset.defaultCharset()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                log.append(line).append(System.lineSeparator());
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IOException("FFmpeg 실행 실패. exitCode=" + exitCode + "\n" + log);
        }

        if (!Files.exists(outputImage)) {
            throw new IOException("출력 이미지가 생성되지 않았습니다.\n" + log);
        }

        return outputImage;
    }

    public static void main(String[] args) throws Exception {
        Path ffmpeg = Path.of("C:\\tools\\ffmpeg\\bin\\ffmpeg.exe");
        Path input  = Path.of("C:\\work\\video\\sample.mp4");
        Path output = Path.of("C:\\work\\video\\thumb\\first.jpg");

        Path result = extractFirstFrame(ffmpeg, input, output);
        System.out.println("생성 완료: " + result);
    }
}