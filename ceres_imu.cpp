#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ceres/ceres.h>
#include <chrono>
#include <vector>

using namespace std;
using namespace ceres;

struct ACC_CostFunction
{
	// constructor
	ACC_CostFunction(double ax, double ay, double az) : ax_(ax), ay_(ay), az_(az) {}
	template <typename T>
	bool operator()(const T *const para_acc, T *residual) const
	{
		T ax_cal = para_acc[0] * ax_ + para_acc[1] * ay_ + para_acc[2] * az_ + para_acc[9];
		T ay_cal = para_acc[3] * ax_ + para_acc[4] * ay_ + para_acc[5] * az_ + para_acc[10];
		T az_cal = para_acc[6] * ax_ + para_acc[7] * ay_ + para_acc[8] * az_ + para_acc[11];
		// define error
		residual[0] = 9.8 * 9.8 - (ax_cal * ax_cal + ay_cal * ay_cal + az_cal * az_cal);

		return true;
	}

private:
	const double ax_, ay_, az_;
};
struct GYRO_CostFunction
{
	// constructor
	GYRO_CostFunction(double wx, double wy, double wz) : wx_(wx), wy_(wy), wz_(wz) {}
	template <typename T>
	bool operator()(const T *const para_gyro, T *residual) const
	{
		T wx_cal = wx_ + para_gyro[0];
		T wy_cal = wy_ + para_gyro[1];
		T wz_cal = wz_ + para_gyro[2];
		// define error
		residual[0] = wx_cal * wx_cal + wy_cal * wy_cal + wz_cal * wz_cal;

		return true;
	}

private:
	const double wx_, wy_, wz_;
};

int main(int argc, char *argv[])
{
	vector<vector<double>> matrix;
	// readfile
	ifstream file;
	if (argc != 2)
	{
		cout << "Using example: " << endl;
		cout << "./Ceres_IMU_Calibration ../imu_data.csv" << endl;
		return 0;
	}
	cout << "----- Fetching .csv file ..." << endl;
	file.open(argv[1]);
	if (!file.is_open())
	{
		cerr << "ERROR: failed to open the file!" << endl;
		return 1;
	}
	string line;
	while (getline(file, line, '\n')) //讀檔讀到跳行字元
	{
		vector<double> row_data;
		stringstream templine(line); // string 轉換成 stream
		string data;
		while (getline(templine, data, ',')) //讀檔讀到逗號
		{
			row_data.push_back(stof(data)); // string 轉換成數字
		}
		matrix.push_back(row_data);
	}
	file.close();
	cout << "----- Finish!" << endl
		 << endl;
	cout << "Whether to show the raw data on the SCREEN? 'y' or 'n': ";
	char resp;
	cin >> resp;
	if (resp == 'y')
	{
		cout << "row_data: " << endl;
		for (int i = 0; i < matrix.size(); i++)
		{
			for (int j = 0; j < matrix[i].size(); j++)
			{
				cout << matrix[i][j] << "  ";
			}
			cout << endl;
		}
	}

	cout << "----- Calling Ceres solver to calibrate the accelerometer" << endl
		 << endl;
	// ceres solver to calcualte acc calibration parameters [s11, s12, s13, s21, s22, s23, s31, s32, s33, bx, by, bz]

	double para_acc[12] = {1.0, 0.0, 0.0,
						   0.0, 1.0, 0.0,
						   0.0, 0.0, 1.0,
						   0.0, 0.0, 0.0};

	Problem problem;

	for (int k = 0; k < matrix.size(); k++)
	{
		CostFunction *cost_function = new AutoDiffCostFunction<ACC_CostFunction, 1, 12>(
			new ACC_CostFunction(matrix[k][0], matrix[k][1], matrix[k][2]));
		problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), para_acc);
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
	cout << "The mathematical model: " << endl;
	cout << "ax_cal = s11 * ax_raw + s12 * ay_raw + s13 * az_raw + bx" << endl
		 << "ay_cal = s21 * ax_raw + s22 * ay_raw + s23 * az_raw + by" << endl
		 << "az_cal = s31 * ax_raw + s32 * ay_raw + s33 * az_raw + bz" << endl
		 << endl;
	cout << "========== Accelerometer calibration parameters output ==========" << endl;
	cout << "=> Scaling matrix: " << endl;
	printf("%10.4f", para_acc[0]);
	printf("%10.4f", para_acc[1]);
	printf("%10.4f\n", para_acc[2]);
	printf("%10.4f", para_acc[3]);
	printf("%10.4f", para_acc[4]);
	printf("%10.4f\n", para_acc[5]);
	printf("%10.4f", para_acc[6]);
	printf("%10.4f", para_acc[7]);
	printf("%10.4f\n", para_acc[8]);
	cout << endl
		 << "=> Bias vector: " << endl;
	printf("%10.4f\n", para_acc[9]);
	printf("%10.4f\n", para_acc[10]);
	printf("%10.4f\n", para_acc[11]);

	cout << endl
		 << endl;
	cout << "----- Calling Ceres solver to calibrate the gyroscope" << endl
		 << endl;
	// ceres solver to calcualte gyro calibration parameters [bx, by, bz]

	double para_gyro[3] = {0.0, 0.0, 0.0};

	for (int k = 0; k < matrix.size(); k++)
	{
		CostFunction *cost_function = new AutoDiffCostFunction<GYRO_CostFunction, 1, 3>(
			new GYRO_CostFunction(matrix[k][3], matrix[k][4], matrix[k][5]));
		problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), para_gyro);
	}
	// using Cauchy function to reduce the influence of the outlier inside the data
	options.linear_solver_type = ceres::DENSE_QR;
	options.max_num_iterations = 50;
	options.minimizer_progress_to_stdout = true;
	Solve(options, &problem, &summary);

	cout << summary.FullReport() << endl;
	cout << "----- Output the result:" << endl;
	cout << "The mathematical model: " << endl;
	cout << "wx_raw + bx" << endl
		 << "wy_raw + by" << endl
		 << "wz_raw + bz" << endl
		 << endl;
	cout << "========== Gyroscope calibration parameters output ==========" << endl;

	cout << "=> Bias vector: " << endl;
	printf("%10.4f\n", para_gyro[0]);
	printf("%10.4f\n", para_gyro[1]);
	printf("%10.4f\n", para_gyro[2]);

	return 0;
}
