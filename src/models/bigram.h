#ifndef BIGRAM_H
#define BIGRAM_H

#include <torch/torch.h>

class Bigram {
public:
  torch::Tensor w1;

  Bigram() {
    torch::Generator g =
        torch::make_generator<at::CPUGeneratorImpl>(2147483647);
    w1 = torch::randn({27, 27}, g, torch::requires_grad(true));
  }

  torch::Tensor forward(torch::Tensor X) {
    torch::Tensor xenc = torch::one_hot(X, 27).to(torch::kFloat32);

    torch::Tensor Z = torch::matmul(xenc, w1);

    return Z;
  }

  torch::Tensor loss(torch::Tensor logits, torch::Tensor Y) {
    return torch::nn::functional::cross_entropy(logits, Y);
  }

  void backward(torch::Tensor loss) { loss.backward(); }

  void grad_des(double lr) {
    torch::NoGradGuard no_grad;

    w1 = w1 - (w1.grad() * lr);
  }
};

#endif
