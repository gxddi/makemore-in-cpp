#ifndef MLP_H
#define MLP_H

#include <ATen/xpu/XPUGeneratorImpl.h>
// #include <math.h>
#include <torch/torch.h>

class MLP {
public:
  // Hyperparams
  int context_len;
  int emb_dim;
  int hid_dim;

  // Layers
  torch::Tensor C;
  torch::Tensor w1;
  torch::Tensor b1;
  torch::Tensor w2;
  torch::Tensor b2;

  // Batch normalization
  torch::Tensor bnmean_running;
  torch::Tensor bnstd_running;
  torch::Tensor bngain;
  torch::Tensor bnbias;

  MLP(int cl, int ed, int hd) : context_len(cl), emb_dim(ed), hid_dim(hd) {
    torch::Generator g =
        torch::make_generator<torch::XPUGeneratorImpl>(2147483647);
    torch::TensorOptions options = torch::device(at::kXPU).requires_grad(true);

    // Layers
    C = torch::randn({27, emb_dim}, g, options);
    w1 = torch::randn({context_len * emb_dim, hid_dim}, g, options);
    b1 = torch::randn({hid_dim}, g, options);
    w2 = torch::randn({hid_dim, 27}, g, options);
    b2 = torch::randn({27}, g, options);
    {
      torch::NoGradGuard no_grad;
      // Kaimin initialization
      w1.mul_((5.0 / 3.0) / sqrt((emb_dim * context_len)));
      b1.mul_(0.01);
      w2.mul_(0.1);
      b2.mul_(0.1);
    }

    // Batch normalization
    bnmean_running = torch::zeros({1, hid_dim}, options);
    bnstd_running = torch::ones({1, hid_dim}, options);
    bngain = torch::ones({1, hid_dim}, options);
    bnbias = torch::zeros({1, hid_dim}, options);
  }

  torch::Tensor forward(torch::Tensor X, bool use_running = false) {
    torch::Tensor xemb =
        torch::matmul(torch::one_hot(X, 27).to(torch::kFloat32), C);
    xemb = xemb.view({-1, context_len * emb_dim});

    // Layer 1
    torch::Tensor Z1 = torch::matmul(xemb, w1) + b1;

    // Batch normalization
    if (!use_running) {
      torch::Tensor bnmeani = Z1.mean({0}, true);
      torch::Tensor bnstdi = Z1.std({0}, true, true);
      Z1 = bngain * ((Z1 - bnmeani) / bnstdi) + bnbias;
      {
        torch::NoGradGuard no_grad;
        bnmean_running = (0.999 * bnmean_running) + (0.001 * bnmeani);
        bnstd_running = (0.999 * bnstd_running) + (0.001 * bnstdi);
      }
    } else {
      Z1 = bngain * ((Z1 - bnmean_running) / bnstd_running) + bnbias;
    }

    // Layer 2
    Z1 = torch::tanh(Z1);
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
    bngain.sub_(bngain.grad() * lr);
    bnbias.sub_(bnbias.grad() * lr);

    C.grad().zero_();
    w1.grad().zero_();
    b1.grad().zero_();
    w2.grad().zero_();
    b2.grad().zero_();
    bngain.grad().zero_();
    bnbias.grad().zero_();
  }
};

#endif
