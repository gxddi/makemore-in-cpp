#include "models/bigram.h"
#include "models/mlp.h"
#include "tools/tools.h"

#include <fstream>
#include <iostream>
#include <string>
#include <torch/torch.h>
#include <vector>

int main() {
  // Get names from ../names.txt
  std::cout << "Getting names...\n";
  std::ifstream file("../names.txt");
  std::vector<std::string> names;
  int nc = get_items(file, names);
  std::vector<std::string> tnames(names.begin(),
                                  names.begin() + (int)(0.8 * nc));
  std::vector<std::string> dnames(names.begin() + (int)(0.8 * nc + 0.1),
                                  names.begin() + (int)0.9 * nc);
  std::vector<std::string> vnames(names.begin() + (int)(0.9 * nc + 0.1),
                                  names.end());

  // Splice names into train/dev/val
  std::cout << "Splicing " << nc << " names...\n";

  int cl = 3; // context length
  std::vector<int> tx, ty, dx, dy, vx, vy;
  get_xy(tnames, cl, tx, ty);
  get_xy(dnames, cl, dy, dy);
  get_xy(vnames, cl, vx, vy);

  torch::Tensor tX = torch::tensor(tx, device(at::kXPU)).view({-1, 3});
  torch::Tensor tY = torch::tensor(ty);

  std::cout << "X (Train):\n" << tX << "\n";
  std::cout << "Y (Train):\n" << tY << "\n";

  // Training
  double lr;

  Bigram simpleBG;
  MLP simpleMLP(cl);

  for (int i = 0; i < 500; i++) {
    torch::Tensor indices = torch::randint(0, tX.size(0), {32});

    torch::Tensor logits = simpleMLP.forward(tX.index_select(0, indices));

    torch::Tensor loss = simpleMLP.loss(logits, tY);

    simpleMLP.backward(loss);

    (i < 1000) ? lr = 0.1 : lr = 0.01;
    simpleMLP.grad_des(lr);

    if (i % 500 == 0)
      std::cout << "Cycle " << i << " (Loss: " << loss << ")";
  }

  // Test w dev set
  torch::Tensor dX = torch::tensor(tx, device(at::kXPU)).view({-1, 3});
  torch::Tensor dY = torch::tensor(ty);
  torch::Tensor logits = simpleMLP.forward(dX);
  torch::Tensor loss = simpleMLP.loss(logits, dY);
  std::cout << "Dev set (Loss: " << loss << ")\n";

  // Generate names
  std::cout << "Generating names...\n";
  torch::Generator g = torch::make_generator<at::CPUGeneratorImpl>(2147483647);
  std::vector<std::string> gen_names = {};

  for (int n = 0; n < 20; n++) {
    std::string name("");
    std::vector<int> context(cl, 0);

    while (true) {
      torch::Tensor input = torch::tensor(context);
      torch::Tensor P = simpleMLP.forward(input).softmax(1);

      int ch_ix = torch::multinomial(P, 1, true, g).item<int>();

      if (ch_ix == 0) {
        break;
      }

      name = name.append(1, (char)ch_ix);

      context.erase(context.begin());
      context.push_back(ch_ix);
    }

    std::cout << name << "\n";
    gen_names.push_back(name);
  }

  return 0;
}
