#ifndef MLP_H
#define MLP_H

#include <ATen/xpu/XPUGeneratorImpl.h>
#include <torch/torch.h>

class MLP {
public:
  torch::Tensor C;
  torch::Tensor w1;
  torch::Tensor b1;
  torch::Tensor w2;
  torch::Tensor b2;
  int context_len;

  MLP(int cl) {
    torch::Generator g =
        torch::make_generator<torch::XPUGeneratorImpl>(2147483647);
    torch::TensorOptions options = torch::device(at::kXPU).requires_grad(true);

    int emb_dim = 10;
    context_len = cl;

    C = torch::randn({27, emb_dim}, g, options);
    w1 = torch::randn({cl * emb_dim, 200}, g, options);
    b1 = torch::randn({200}, g, options);
    w2 = torch::randn({200, 27}, g, options);
    b2 = torch::randn({27}, g, options);

    torch::NoGradGuard no_grad;
    C.mul_(0.1);
    w1.mul_(0.1);
    b1.mul_(0.1);
    w2.mul_(0.1);
    b2.mul_(0.1);
  }

  torch::Tensor forward(torch::Tensor X) {
    torch::Tensor xemb =
        torch::matmul(torch::one_hot(X, 27).to(torch::kFloat32), C);
    xemb = xemb.view({-1, context_len * C.size(1)});

    torch::Tensor Z1 = torch::tanh(torch::matmul(xemb, w1) + b1);
    torch::Tensor Z2 = torch::matmul(Z1, w2) + b2;

    return Z2;
  }

  torch::Tensor loss(torch::Tensor logits, torch::Tensor Y) {
    return torch::nn::functional::cross_entropy(logits, Y);
  }

  void backward(torch::Tensor loss) { loss.backward(); }

  void grad_des(double lr) {
    torch::NoGradGuard no_grad;

    C.sub_(C.grad() * lr);
    w1.sub_(w1.grad() * lr);
    b1.sub_(b1.grad() * lr);
    w2.sub_(w2.grad() * lr);
    b2.sub_(b2.grad() * lr);

    C.grad().zero_();
    w1.grad().zero_();
    b1.grad().zero_();
    w2.grad().zero_();
    b2.grad().zero_();
  }
};

#endif
