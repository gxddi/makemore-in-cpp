#include "data/load.hpp"
#include "models/bigram.h"
#include "models/mlp.h"

#include <ATen/xpu/XPUGeneratorImpl.h>
#include <torch/torch.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main() {
  // HYPERPARAMETERS
  int cl = 5;       // Context len
  int emb_dim = 10; // Embedding dimension
  int hl = 200;     // Number neurons in the hidden layer

  // Get names from ../names.txt and shuffle
  std::cout << "Getting names...\n";
  std::ifstream file("../data/names.txt"); // input filebuf stream
  std::vector<std::string> names;
  load_items(file, names);
  std::mt19937 r(42); // shuffle names
  std::shuffle(names.begin(), names.end(), r);

  // Split names into train (80%) / dev (10%) / val (10%)
  std::cout << "Splicing " << names.size() << " names...\n";
  std::vector<std::string> tnames(names.begin(),
                                  names.begin() + (int)(0.8 * names.size()));
  std::vector<std::string> dnames(names.begin() +
                                      (int)((0.8 * names.size()) + 1),
                                  names.begin() + (int)(0.9 * names.size()));
  std::vector<std::string> vnames(
      names.begin() + (int)((0.9 * names.size()) + 1), names.end());

  std::vector<int> tx, ty, dx, dy, vx, vy;
  load_dataset(tnames, cl, tx, ty);
  load_dataset(dnames, cl, dx, dy);
  load_dataset(vnames, cl, vx, vy);

  torch::Tensor tX = torch::tensor(tx, device(at::kXPU)).view({-1, cl});
  torch::Tensor tY = torch::tensor(ty, device(at::kXPU));
  torch::Tensor dX = torch::tensor(dx, device(at::kXPU)).view({-1, cl});
  torch::Tensor dY = torch::tensor(dy, device(at::kXPU));
  torch::Tensor vX = torch::tensor(vx, device(at::kXPU)).view({-1, cl});
  torch::Tensor vY = torch::tensor(vy, device(at::kXPU));

  // Training
  std::cout << "Training...\n";

  // Bigram simpleBG;
  MLP simpleMLP(cl, emb_dim, hl);

  double lr;
  for (int i = 0; i < 200000; i++) {
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
      torch::Tensor logits_t = simpleMLP.forward(tX);
      torch::Tensor loss_t = simpleMLP.loss(logits_t, tY);
      std::cout << "        (Training loss: " << loss_t.item() << ")\n";

      // Test w dev set
      torch::Tensor logits_d = simpleMLP.forward(dX);
      torch::Tensor loss_d = simpleMLP.loss(logits_d, dY);
      std::cout << "        (Dev loss: " << loss_d.item() << ")\n";
    }
  }

  // Final dev set test
  torch::Tensor logits_d = simpleMLP.forward(dX, true);
  torch::Tensor loss_d = simpleMLP.loss(logits_d, dY);
  std::cout << "Final dev set loss: " << loss_d.item() << "\n";

  // Final val set loss
  torch::Tensor logits_v = simpleMLP.forward(vX, true);
  torch::Tensor loss_v = simpleMLP.loss(logits_v, vY);
  std::cout << "Final val set loss: " << loss_v.item() << "\n";

  // Generate names with model
  std::cout << "Generating names...\n";

  torch::Generator g = torch::make_generator<at::XPUGeneratorImpl>(2147483647);
  std::vector<std::string> gen_names;

  for (int n = 0; n < 20; n++) { // 20 names
    std::string name("");
    std::vector<int> context(cl, 0);

    while (true) {
      torch::Tensor input = torch::tensor(context, device(at::kXPU));
      torch::Tensor P = simpleMLP.forward(input, true).softmax(1);

      int ch_ix = torch::multinomial(P, 1, false, g).item<int>();

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
