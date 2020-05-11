#include <torch/extension.h>
#include <ATen/ParallelNative.h>

#include <iostream>

torch::Tensor sparse_matmul(torch::Tensor& z, torch::Tensor& d) {
//  std::cout << "z size: " << z.sizes() << std::endl;
//  std::cout << "d size: " << d.sizes() << std::endl;

  long int batchSize = z.sizes()[0];
  std::vector<torch::Tensor> mStor;
  mStor.resize(batchSize);

  for(long int i=0; i<batchSize; i++) {
    mStor[i] = torch::_sparse_mm(z[i],d[i]);
  }

//  torch::Tensor resultMat = torch::stack(mStor);
//  torch::Tensor resultMat = torch::zeros({batchSize, z.sizes()[1], d.sizes()[2]}, options);
//  torch::Tensor resultMat = torch::unsqueeze(torch::_sparse_mm(z[0], d[0]), 0);
//    resultMat = torch::cat({resultMat, torch::unsqueeze(mStor[i], 0)});

//  std::cout << resultMat.sizes() << std::endl;
  
  return torch::stack(mStor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_matmul", &sparse_matmul, "3D tensor sparse matmul");
}
