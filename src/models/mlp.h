#ifndef MLP_H
#define MLP_H

#include <torch/torch.h>

class MLP {
public:
  torch::Tensor C;
  torch::Tensor w1;
  torch::Tensor b1;
  torch::Tensor w2;
  torch::Tensor b2;
  int context_len;

  MLP(int context_len) {
    torch::Generator g =
        torch::make_generator<torch::CPUGeneratorImpl>(2147483647);
    torch::TensorOptions options = torch::device(at::kXPU).requires_grad(true);

    int emb_dim = 10;

    this->context_len = context_len;
    C = torch::randn({27, emb_dim}, g, options);
    w1 = torch::randn({context_len * emb_dim, 300}, g, options);
    b1 = torch::randn({300}, g, options);
    w2 = torch::randn({300, 27}, g, options);
    b2 = torch::randn({27}, g, options);
  }

  torch::Tensor forward(torch::Tensor X) {
    torch::Tensor xemb =
        torch::matmul(torch::one_hot(X, 27).to(torch::kFloat32), C);
    xemb = xemb.view({-1, (context_len * C.size(1))});

    torch::Tensor Z1 = torch::tanh(torch::matmul(xemb, w1) + b1);
    torch::Tensor Z2 = torch::matmul(Z1, w2) + b2;

    return Z2;
  }

  torch::Tensor loss(torch::Tensor logits, torch::Tensor Y) {
    return torch::nn::functional::cross_entropy(logits, Y);
  }

  torch::Tensor backward(torch::Tensor loss) { loss.backward(); }

  torch::Tensor grad_des(double lr) {
    torch::NoGradGuard no_grad;

    w1 = w1 - (w1.grad() * lr);
  }
};

#endif
