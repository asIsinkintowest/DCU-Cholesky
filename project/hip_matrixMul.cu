#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <hip/hip_runtime.h>
using namespace std;

#define WIDTH 32

__global__ void Matrix_Mul_Kernel(float *M, float *N, float *P, int width)
{
    // 计算输出元素的位置
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float pValue = 0.0f;

    for (int k = 0; k < width; ++k)
    {
        pValue += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = pValue;
}

float *read_csv(const string &filename)
{
    ifstream file(filename); // 打开文件系统
    string line;
    vector<float> values;

    while (getline(file, line))
    {
        stringstream strstream(line); // 把字符串包装成流
        string value;

        while (getline(strstream, value, ','))
        {
            values.push_back(stof(value));
        }
    }

    float *arr = new float[values.size()];
    memcpy(arr, values.data(), values.size() * sizeof(float));
    return arr;
}

void save_csv(const string filename, float *P, int width)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "cant open file\n"
             << filename << endl;
        return;
    }

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            file << P[i * width + j];
            if (j < width - 1)
            {
                file << ',';
            }
        }
        file << '\n';
    }

    file.close();
}

int main(void)
{
    // 读取输入矩阵
    float *M_h = read_csv("data/matrix1.csv");
    float *N_h = read_csv("data/matrix2.csv");

    // 分配设备端内存
    float *M_d, *N_d, *P_d;
    hipMalloc(&M_d, sizeof(float) * WIDTH * WIDTH);
    hipMalloc(&N_d, sizeof(float) * WIDTH * WIDTH);
    hipMalloc(&P_d, sizeof(float) * WIDTH * WIDTH);

    // 提前将数据传过去
    hipMemcpy(M_d, M_h, sizeof(float) * WIDTH * WIDTH, hipMemcpyHostToDevice);
    hipMemcpy(N_d, N_h, sizeof(float) * WIDTH * WIDTH, hipMemcpyHostToDevice);

    // 配置Kernel参数
    dim3 block(WIDTH, WIDTH);
    dim3 grid((WIDTH + block.x - 1) / block.x, (WIDTH + block.y - 1) / block.y);

    // 计算
    Matrix_Mul_Kernel<<<grid, block>>>(M_d, N_d, P_d, WIDTH);
    hipDeviceSynchronize();

    float *P_h = new float[WIDTH * WIDTH];
    hipMemcpy(P_h, P_d, sizeof(float) * WIDTH * WIDTH, hipMemcpyDeviceToHost);

    save_csv("./data/output2.csv", P_h, WIDTH);

    hipFree(M_d);
    hipFree(N_d);
    hipFree(P_d);
    delete[] M_h;
    delete[] N_h;
    delete[] P_h;

    return 0;
}