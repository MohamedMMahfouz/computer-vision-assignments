import org.ietf.jgss.Oid;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Ass1 {

	 
	public static Mat Median_filter(Mat src){
		
		//Mat src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
		Mat dst = new Mat();
		if( src.empty() ) {
            System.out.println("Error ");
            System.exit(-1);
        }
		int Kernel = 5;
		int delay = 1;
		
            Imgproc.medianBlur(src, dst, Kernel);
            HighGui.imshow( "Median filter", dst );
	        int c = HighGui.waitKey( delay );  	
		return dst;
	}
	
	public static Mat Sharpen(Mat src){
		// sharpen image using "unsharp mask" algorithm
		//Mat src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
			         Mat smoothed = new Mat(src.rows(),src.cols(),src.type());
			         int MAX_KERNEL_LENGTH = 31;
//			         for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
			             Imgproc.blur(src, smoothed, new Size(3, 3));
			             
//			         }
			         Core.addWeighted(src, 1.5, smoothed, -0.5, 0, smoothed);
			         /////////getting Edges(X,Y)
			         Mat blurredImage =smoothed;
			         Mat originalImage = Imgcodecs.imread("1.jpg", Imgcodecs.IMREAD_COLOR);
			         Mat img = Mat.zeros(blurredImage.size(), blurredImage.type());
			          
			         //blurred image data
			         byte[] blurredImageData = new byte[(int) (blurredImage.total()*blurredImage.channels())];
			         blurredImage.get(0, 0, blurredImageData);
			         //original image data
			         byte[] originalImageData = new byte[(int) (originalImage.total()*originalImage.channels())];
			         originalImage.get(0, 0, originalImageData);
			         
			         byte[] smoothedImageData = new byte[(int) (img.total()*img.channels())];
			         byte[] newImageData = new byte[(int) (img.total()*img.channels())];
			         
			         ////////////edges
//			      
			         for (int i = 0; i < blurredImage.rows(); i++) {
			             for (int j = 0; j < blurredImage.cols(); j++) {
			                 for (int k = 0;k < blurredImage.channels(); k++) {
			                     double PValue =(originalImageData[(i * originalImage.cols() + j) * originalImage.channels() + k]-  blurredImageData[(i * blurredImage.cols() + j) * blurredImage.channels() + k] );
			                     PValue = PValue < 0 ? PValue + 256 : PValue;
			                     int iVal = (int) Math.round(PValue);
			                     iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
			                     smoothedImageData[(i * originalImage.cols() + j) * originalImage.channels() + k] = (byte) iVal;
			                 }
			             }
			         }
			         
			     Mat EdgeImage = Mat.zeros(blurredImage.size(), blurredImage.type());
			     EdgeImage.put(0, 0, smoothedImageData);
			     byte[] EdgeImageData = new byte[(int) (EdgeImage.total()*EdgeImage.channels())];
		         EdgeImage.get(0, 0, EdgeImageData);
			     
		         /////////unsharp
			      
			         for (int i = 0; i < blurredImage.rows(); i++) {
			             for (int j = 0; j < blurredImage.cols(); j++) {
			                 for (int k = 0;k < blurredImage.channels(); k++) {
			                     double PValue = (originalImageData[(i * originalImage.cols() + j) * originalImage.channels() + k]+ (0.2*(blurredImageData[(i * blurredImage.cols() + j) * blurredImage.channels() + k])) );
			                     PValue = PValue < 0 ? PValue + 256 : PValue;
			                     int iVal = (int) Math.round(PValue);
			                     iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
			                     newImageData[(i * blurredImage.cols() + j) * blurredImage.channels() + k] = (byte) iVal;
			                 }
			             }
			         }
			                 
			         
			         img.put(0, 0, newImageData);
			         HighGui.imshow("sharp.jpg", img);
			         HighGui.waitKey( 1 );
		         return img;
		         
	}
	
	
	public static Mat Contrast( Mat image) {
        
        Mat img = Mat.zeros(image.size(), image.type());
        double alpha = 2.5; 
        int beta = 1;  
        byte[] imgData = new byte[(int) (image.total()*image.channels())];
        image.get(0, 0, imgData);
        byte[] newImageData = new byte[(int) (img.total()*img.channels())];
        for (int i = 0; i < image.rows(); i++) {
            for (int j = 0; j < image.cols(); j++) {
                for (int k = 0;k < image.channels(); k++) {
                    double PValue = imgData[(i * image.cols() + j) * image.channels() + k];
                    PValue = PValue < 0 ? PValue + 256 : PValue;
                    int iVal = (int) Math.round(alpha * PValue + beta);
                    iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
                    newImageData[(i * image.cols() + j) * image.channels() + k] = (byte) iVal;
                }
            }
        }
     
        img.put(0, 0, newImageData);
        
        return img;
    }

	
    public static Mat Contrast2(Mat src){
    	 //Mat src = Imgcodecs.imread(filename);
    	Mat LUT = new Mat(1, 256, CvType.CV_8U);
         byte[] lookUpTableData = new byte[(int) (LUT.total()*LUT.channels())];
         for (int i = 0; i < LUT.cols(); i++) {
        	 int iVal = (int) Math.round(Math.pow(i / 255.0, 3.3) * 255.0);
             iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
        	 lookUpTableData[i] = (byte)iVal;
         }
         LUT.put(0, 0, lookUpTableData);
         Mat img = new Mat();
         Core.LUT(src, LUT, img); 
         Mat dst = Contrast(img);
         HighGui.imshow("FINAL IMAGE", dst);
         HighGui.waitKey();
         
         return dst;
    }


    private static double Vairance(String filename) {
    	
    	Mat src = Imgcodecs.imread(filename);
    	
    	MatOfDouble mean = new MatOfDouble();
    	MatOfDouble std = new MatOfDouble();
    	Core.meanStdDev(src, mean, std);
    	double[] means = mean.get(0, 0);
    	double[] stds = std.get(0, 0);
    	double score = 0.0;
    	for (int i = 0; i < means.length; i++) {
    		score += means[i] - stds[i];
    	}
    	//score*=score;
    	double Variance = stds[0];
    	Variance*=Variance;
    	System.out.println(Variance);
    	return score;
    }

    public static void measureBlur(String filename){
    	
    ///create the image with edges
        Mat grad = new Mat();
        String window_name = "Sobel Demo - Simple Edge Detector";
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
      
        Mat src = Imgcodecs.imread(filename);
   
    
        Mat grad_x = new Mat(), grad_y = new Mat();
        Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

        Imgproc.Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, Core.BORDER_DEFAULT );

        Imgproc.Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, Core.BORDER_DEFAULT );
        // converting back to CV_8U
        Core.convertScaleAbs( grad_x, abs_grad_x );
        Core.convertScaleAbs( grad_y, abs_grad_y );
        Core.addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        //System.out.println(grad. +" "+abs_grad_y.toString() );
        HighGui.imshow( window_name, grad );
        HighGui.waitKey(0);
        System.exit(0);
    }
    
    public static double getMean(double[][]x ){
		double Mean = 0 ;
		for(int i = 0 ; i<x.length ; i++){
			for (int j = 0 ; j<x[i].length ; j++){
				Mean += x[i][j];
			}
		}
		return Mean/(25);
	}
	
	public static double getVariance(double[][]x , double mean){
		
		
		double variance = 0 ;
		
		
		for(int i = 0 ; i<x.length ; i++){
			for (int j = 0 ; j<x[i].length ; j++){
				variance += Math.pow(x[i][j]-mean, 2);
			}
		}
		return variance /25;
	}
	
	
	public static boolean DetectNoisyImage(Mat img){
		int noisyPx = 0 ; 
		for(int i = 1 ; i <img.rows()-4;i++){
			double mean = 0;
			double variance = 0;
			for(int j = 1 ; j<img.cols()-4 ; j++){
				Mat currentMatrix = new Mat();
				double[][] matrix = new double[5][5];
				matrix[0][0] = img.get(i, j)[0];
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
				
				mean = getMean(matrix);		
				variance = getVariance(matrix,mean);
				//System.out.print("Mean: " + mean + " ");
				//System.out.println("Variance: " + variance);
				if(variance>1000){
					noisyPx ++;
				}
			}
		}
		double Percentage= (noisyPx/(1080.0*1920.0))*100;
	System.out.println("noisy% = "+Percentage);	
	if(noisyPx > (0.4*1080*1920)){
			return true;
		}
		return false;
	}
	
	public static double multiplyKernel(double [][]m1 , double [][]m2){
		
		double sum = 0 ;
		for(int i = 0 ; i<m1.length ; i++){
			for (int j = 0; j < m1[i].length; j++) {
				sum+= m1[i][j] *m2[i][j];
			}
		}
		return sum;
	}
	
	public static boolean detectBlurry(Mat img){
		double[][] gxMatrix = {{-1,0,1},{-2,0,2},{-1,0,1}};
		double[][] gyMatrix = {{1,2,1},{0,0,0},{-1,-2,-1}};
		Mat out = new Mat(img.rows(),img.cols(),img.type());
		double [][] output = new double[img.rows()][img.cols()];
		double gx , gy , gTotal = 0;
		int blurryPx = 0 ;
		for(int i = 1 ; i<img.rows()-1 ; i++){
			for(int j = 1 ; j<img.cols()-1 ; j++){
				 double [][]matrix = {{img.get(i-1, j-1)[0],img.get(i-1, j)[0],img.get(i-1, j+1)[0]},{img.get(i, j-1)[0],img.get(i, j)[0],img.get(i, j+1)[0]},
						{img.get(i+1, j-1)[0],img.get(i+1, j)[0],img.get(i+1, j+1)[0],}};
				 gx = multiplyKernel(matrix, gxMatrix);
				 gy = multiplyKernel(matrix, gyMatrix);
				 gTotal = Math.abs(gx) + Math.abs(gy); 
				//System.out.println(gTotal);
				 if(gTotal<10){
					 blurryPx++;
				}
				output[i][j] = gTotal;
			}
		}
		double Percentage= (blurryPx/(1080.0*1920.0))*100;
		System.out.println("blurry% = " + Percentage);
		if(blurryPx > (0.25*1920*1080))
			return true;
		
		return false;
	}
	
	public static int[] createHistogram(Mat img){
		int []histogram = new int[256];
		for(int i = 0 ; i<img.rows();i++){
			for(int j = 0 ; j<img.cols();j++){
				int current = (int)img.get(i, j)[0];
				histogram[current]++;
			}
		}
		return histogram;
	}
	public static int getMinIntensity(int[]x){
		int min = -1 ;
	for (int i = 0; i < x.length; i++) {
		if (x[i]> 0) 
		return i;
	}	
	return min;
	}
	public static int getMaxIntensity(int[]x){
		int max = -1 ;
		for (int i = x.length-1; i > 0; i--) {
			if(x[i]>0)
				return i;
		}
		return max ;
	}
	
	public static double getPercentageColorUsed(int[]x){
		double count = 0 ;
		
		for(int i =0 ; i<x.length ; i++){
			if(x[i]>0)
				count++;
		}
		return count/256.0 ;
		
	}
	public static boolean detectColorCollapsing(Mat img){
		int [] histogram = createHistogram(img);
		double percentage = getPercentageColorUsed(histogram);
		
		System.out.println( "Max intensity: "+getMaxIntensity(histogram));
		System.out.println( "Min intensity: "+getMinIntensity(histogram));
		System.out.println("color collapsing% = "+ percentage*100);
		if(percentage<1.0)
			return true;
		return false;
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	
		String filename = "4.jpg";
		Mat img = Imgcodecs.imread(filename);

		////////////measure noise
		boolean noisy = DetectNoisyImage(img);
		System.out.println("Noisy = "+ noisy);
		//////correct noise
		if(noisy){
			img = Median_filter(img);
			
		}
		
		//////////////measure blurry
		boolean blurry = detectBlurry(img);
		boolean collapsed = detectColorCollapsing(img);
		System.out.println("Blurry= "+ blurry);
		////////correct blurry
		if(blurry){
			img = Sharpen(img);
		}
		
		///////////////measure color collapsing
		//boolean collapsed = detectColorCollapsing(img);
		System.out.println("collapsed= "+ collapsed);
		//////////correct color collapsing
		if(collapsed){
			img = Contrast2(img);
		}
		///print final image
		HighGui.imshow("FINAL IMAGE", img);
        HighGui.waitKey();
////		
		
		
		
		
		
		
		
		
		
	}
	
}
