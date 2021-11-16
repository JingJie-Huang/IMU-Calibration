#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <opencv2/core/core.hpp>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <cmath>
#include <chrono>
#include <vector>


using namespace std;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex: public g2o::BaseVertex<6, Eigen::Matrix<double,6,1>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl()
    {
		_estimate << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
	}
    
    virtual void oplusImpl( const double* update )
    {
        _estimate += Eigen::Matrix<double,6,1> (update);
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) { return false; }
    virtual bool write( ostream& out ) const { return false; }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double ax, double ay, double az ): BaseUnaryEdge(), ax_(ax), ay_(ay), az_(az) {}
    // 计算曲线模型误差
    virtual void computeError()
    {
    	// para = [sx, sy, sz, bx, by, bz]
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Matrix<double,6,1> para = v->estimate();
        //cout<<para(0,0)<<", "<<para(1,0)<<", "<<para(2,0)<<", "<<para(3,0)<<", "<<para(4,0)<<", "<<para(5,0)<<endl;
        _error(0,0) = _measurement - ( (para(0,0) * (ax_ + para(3,0))) * (para(0,0) * (ax_ + para(3,0))) +
									   (para(1,0) * (ay_ + para(4,0))) * (para(1,0) * (ay_ + para(4,0))) +
									   (para(2,0) * (az_ + para(5,0))) * (para(2,0) * (az_ + para(5,0)))
   									 );
		// cout << _error(0,0) << endl;
    }
    
    virtual void linearizeOplus()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Matrix<double,6,1> para = v->estimate();
        _jacobianOplusXi[0] = -2.0*para(0,0)*( (ax_ + para(3,0)) * (ax_ + para(3,0)) );
        _jacobianOplusXi[1] = -2.0*para(1,0)*( (ay_ + para(4,0)) * (ay_ + para(4,0)) );
        _jacobianOplusXi[2] = -2.0*para(2,0)*( (az_ + para(5,0)) * (az_ + para(5,0)) );
        _jacobianOplusXi[3] = -2.0*para(0,0)*para(0,0)*(ax_ + para(3,0));
        _jacobianOplusXi[4] = -2.0*para(1,0)*para(1,0)*(ay_ + para(4,0));
        _jacobianOplusXi[5] = -2.0*para(2,0)*para(2,0)*(az_ + para(5,0));
        
    }
    
    
    virtual bool read( istream& in ) { return false; }
    virtual bool write( ostream& out ) const { return false; }
public:
    const double ax_, ay_, az_;  // acc 值， 9.8^2 值为 _measurement
};

int main( int argc, char** argv )
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

	// number of rows of the raw data
	int N = matrix.size();
    
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,1> > Block;  // 每个误差项优化变量维度为6，误差值维度为1
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器
    // 梯度下降方法，从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm( solver );   // 设置求解器
    optimizer.setVerbose( true );       // 打开调试输出
    
    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
	Eigen::Matrix<double,6,1> para_initial;
	//Vector6d << 1.05, 0.95, 0.98, 1.5, -1.2, 0.7;
    para_initial << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
	v->setEstimate( para_initial );
    v->setId(0);
    optimizer.addVertex( v );
    
    // 往图中增加边
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( matrix[i][0], matrix[i][1], matrix[i][2] );
        edge->setId(i);
        edge->setVertex( 0, v );                // 设置连接的顶点
        edge->setMeasurement( 9.8*9.8 );      // 观测数值
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity() ); // 信息矩阵：协方差矩阵之逆
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;  // Huber function (kill outlier data)
        // rk->setDelta(1.0);
        cout << rk->delta() << endl;
        edge->setRobustKernel(rk);
        optimizer.addEdge( edge );
    }
    
    // 执行优化
    cout<<"start optimization"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(50);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    
    // 输出优化值
    Eigen::Matrix<double,6,1> para_estimate = v->estimate();
    cout << "----- Output the result:" << endl;
	cout << "	The mathematical model: " << endl;
	cout << "	ax_cal = sx * (ax_raw + ba)" << endl
		 << "	ay_cal = sy * (ay_raw + by)" << endl
		 << "	az_cal = sz * (az_raw + bz)" << endl << endl;
	cout << "	sx = " << para_estimate(0,0) << endl;
	cout << "	sy = " << para_estimate(1,0) << endl;
	cout << "	sz = " << para_estimate(2,0) << endl;
	cout << "	bx = " << para_estimate(3,0) << endl;
	cout << "	by = " << para_estimate(4,0) << endl;
	cout << "	bz = " << para_estimate(5,0) << endl;
    
    return 0;
}






