#include "models/bigram.h"
#include "models/mlp.h"
#include "tools/tools.h"

#include <ATen/xpu/XPUGeneratorImpl.h>
#include <torch/torch.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
  std::cout << torch::xpu::is_available() << "\n";
  // Get names from ../names.txt
  std::cout << "Getting names...\n";
  std::ifstream file("../names.txt");
  std::vector<std::string> names;
  get_items(file, names);
  std::mt19937 g(42); // shuffle names
  std::shuffle(names.begin(), names.end(), g);

  // Split names into train/dev/val
  std::vector<std::string> tnames(names.begin(),
                                  names.begin() + (int)(0.8 * names.size()));
  std::vector<std::string> dnames(names.begin() +
                                      (int)((0.8 * names.size()) + 1),
                                  names.begin() + (int)(0.9 * names.size()));
  std::vector<std::string> vnames(
      names.begin() + (int)((0.9 * names.size()) + 1), names.end());

  // std::cout << "Training names:\n" << tnames << "\n";
  // std::cout << "Dev names:\n" << dnames << "\n";

  std::cout << "Splicing " << names.size() << " names...\n";

  int cl = 5; // context length
  std::vector<int> tx, ty, dx, dy, vx, vy;
  get_xy(tnames, cl, tx, ty);
  get_xy(dnames, cl, dx, dy);
  get_xy(vnames, cl, vx, vy);

  torch::Tensor tX = torch::tensor(tx, device(at::kXPU)).view({-1, cl});
  torch::Tensor tY = torch::tensor(ty, device(at::kXPU));
  torch::Tensor dX = torch::tensor(dx, device(at::kXPU)).view({-1, cl});
  torch::Tensor dY = torch::tensor(dy, device(at::kXPU));
  // std::cout << "X (dev):\n" << dx << "\n";
  // std::cout << "Y (dev):\n" << dy << "\n";

  // Training
  std::cout << "Training...\n";
  double lr;
  torch::manual_seed(2147483647);
  // Bigram simpleBG;
  MLP simpleMLP(cl);

  for (int i = 0; i < 300000; i++) {
    torch::Tensor indices =
        torch::randint(0, tX.size(0), {32}, device(at::kXPU).dtype(at::kInt));

    torch::Tensor logits = simpleMLP.forward(tX.index_select(0, indices));

    torch::Tensor loss = simpleMLP.loss(logits, tY.index_select(0, indices));

    simpleMLP.backward(loss);

    (i < 100000) ? lr = 0.1 : lr = 0.01;
    simpleMLP.grad_des(lr);

    if (i % 10000 == 0) {
      std::cout << "Cycle " << i << " (Batch loss: " << loss.item() << ")\n";
      // Test w train set
      torch::Tensor logits = simpleMLP.forward(tX);
      torch::Tensor loss = simpleMLP.loss(logits, tY);
      std::cout << "        (Training loss: " << loss.item() << ")\n";

      // Test w dev set
      logits = simpleMLP.forward(dX);
      loss = simpleMLP.loss(logits, dY);
      std::cout << "        (Dev loss: " << loss.item() << ")\n";
    }
  }

  // Final dev set test
  torch::Tensor logits = simpleMLP.forward(dX);
  torch::Tensor loss = simpleMLP.loss(logits, dY);
  std::cout << "Final dev set loss: " << loss.item() << "\n";

  // Generate names
  std::cout << "Generating names...\n";
  // torch::Generator g =
  // torch::make_generator<at::XPUGeneratorImpl>(2147483647);
  std::vector<std::string> gen_names = {};

  for (int n = 0; n < 20; n++) {
    std::string name("");
    std::vector<int> context(cl, 0);

    while (true) {
      torch::Tensor input = torch::tensor(context, device(at::kXPU));
      torch::Tensor P = simpleMLP.forward(input).softmax(1);

      int ch_ix = torch::multinomial(P, 1, true).item<int>();

      if (ch_ix == 0) {
        break;
      }

      name = name.append(1, (char)(ch_ix + 96));

      context.erase(context.begin());
      context.push_back(ch_ix);
    }

    std::cout << name << "\n";
    gen_names.push_back(name);
  }

  return 0;
}
