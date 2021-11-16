#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<ceres/ceres.h>
#include<chrono>
#include<vector>


using namespace std;
using namespace ceres;


struct IMU_CostFunction{
	// constructor
	IMU_CostFunction(double ax, double ay, double az): ax_(ax), ay_(ay), az_(az) {}
	template <typename T>
	bool operator ()(const T *const para, T *residual) const {
		//error
		residual[0] =  9.8*9.8 - (  (para[0]*(ax_ + para[3])) * (para[0]*(ax_ + para[3])) + 
								    (para[1]*(ay_ + para[4])) * (para[1]*(ay_ + para[4])) + 
					                (para[2]*(az_ + para[5])) * (para[2]*(az_ + para[5]))  );
		return true;
	}
	private:
	const double ax_, ay_, az_;
};


int main(int argc, char *argv[])
{
	vector<vector<double>> matrix;
	//readfile
	ifstream file;
	cout << "----- Fetching .csv file ..." << endl;
	file.open("../imu_data.csv");
	if(!file.is_open()){
		cerr << "ERROR: failed to open the file!" << endl;
		return 1;
	}
	string line;
	while (getline( file, line,'\n'))  //讀檔讀到跳行字元
	{
		vector<double> row_data;
		stringstream templine(line); // string 轉換成 stream
		string data;
		 while (getline( templine, data,',')) //讀檔讀到逗號
		 {
			 row_data.push_back(stof(data));  //string 轉換成數字
		 }
		matrix.push_back(row_data);
	}
	file.close();
	cout << "----- Finish!" << endl << endl;
	cout << "Whether to show the raw data on the SCREEN? 'y' or 'n': ";
	char resp;
	cin >> resp;
	if(resp == 'y'){
		cout << "row_data: " << endl;
		for(int i=0; i<matrix.size(); i++){
			for(int j=0; j<matrix[i].size(); j++){
				cout << matrix[i][j] << "  ";
			}
			cout << endl;
		}
	}

	cout << "----- Calling Ceres solver to calibrate the imu data" << endl << endl;
	// ceres solver to calcualte imu calibration parameters [sx, sy, sz, bx, by, bz]

	double para[6] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
	
	Problem problem;

	for(int k=0; k<matrix.size(); k++){
		CostFunction *cost_function = new AutoDiffCostFunction<IMU_CostFunction,1,6>(
		new IMU_CostFunction(matrix[k][0], matrix[k][1], matrix[k][2])  );
		problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), para);
	}
	// using Cauchy function to reduce the influence of the outlier inside the data

	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.max_num_iterations = 50;
	options.minimizer_progress_to_stdout = true;
	Solver::Summary summary;
	Solve(options, &problem, &summary);

	cout << summary.FullReport() << endl;
	cout << "----- Output the result:" << endl;
	cout << "	The mathematical model: " << endl;
	cout << "	ax_cal = sx * (ax_raw + ba)" << endl
		 << "	ay_cal = sy * (ay_raw + by)" << endl
		 << "	az_cal = sz * (az_raw + bz)" << endl << endl;
	cout << "	sx = " << para[0] << endl;
	cout << "	sy = " << para[1] << endl;
	cout << "	sz = " << para[2] << endl;
	cout << "	bx = " << para[3] << endl;
	cout << "	by = " << para[4] << endl;
	cout << "	bz = " << para[5] << endl;
	

	return 0;
}






