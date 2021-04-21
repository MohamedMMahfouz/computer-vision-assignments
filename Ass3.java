import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Ass3 {

	
	public static boolean DisparityEstimation(Mat img){
		 
		for(int i = 1 ; i <img.rows()-4;i++){		
			for(int j = 1 ; j<img.cols()-4 ; j++){
				Mat currentMatrix = new Mat();
				double[][] matrix = new double[5][5];
				matrix[0][0] = img.get(i, j)[3];
				matrix[0][1] = img.get(i, j+1)[0];
				matrix[0][2] = img.get(i, j+2)[0];
				matrix[0][3] = img.get(i, j+3)[0];
				matrix[0][4] = img.get(i, j+4)[0];
				matrix[1][0] = img.get(i+1, j)[0];
				matrix[1][1] = img.get(i+1, j+1)[0];
				matrix[1][2] = img.get(i+1, j+2)[0];
				matrix[1][3] = img.get(i+1, j+3)[0];
				matrix[1][4] = img.get(i+1, j+4)[0];
				matrix[2][0] = img.get(i+2, j)[0];
				matrix[2][1] = img.get(i+2, j+1)[0];
				matrix[2][2] = img.get(i+2, j+2)[0];
				matrix[2][3] = img.get(i+2, j+3)[0];
				matrix[2][4] = img.get(i+2, j+4)[0];
				matrix[3][0] = img.get(i+3, j)[0];
				matrix[3][1] = img.get(i+3, j+1)[0];
				matrix[3][2] = img.get(i+3, j+2)[0];
				matrix[3][3] = img.get(i+3, j+3)[0];
				matrix[3][4] = img.get(i+3, j+4)[0];
				matrix[4][0] = img.get(i+4, j)[0];
				matrix[4][1] = img.get(i+4, j+1)[0];
				matrix[4][2] = img.get(i+4, j+2)[0];
				matrix[4][3] = img.get(i+4, j+3)[0];
				matrix[4][4] = img.get(i+4, j+4)[0];
				
				
			}
		}
		
		return false;
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Mat img = Imgcodecs.imread("As3.jpg",Imgcodecs.IMREAD_COLOR);
		DisparityEstimation(img);
	}
}
