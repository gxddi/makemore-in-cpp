#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

int get_names(std::ifstream *file, std::vector<std::string> *names) {
  int nc = 0; // name count
  std::string line;
  while (std::getline(*file, line)) {
    (*names).push_back(line);
    nc++;
  }

  return nc;
}

torch::Tensor get_bigram_counts(std::string *names, int nc) {
  torch::Tensor N = torch::zeros({27, 27});

  for (int nix = 0; nix < nc; nix++) {
    std::string name = names[nix];
    for (int cix = 0; cix < (name.size() + 1); cix++) {
      int ich1;
      int ich2;

      if (cix == 0) {
        ich1 = 0;
      } else {
        ich1 = (int)name[cix - 1] - 96;
      }

      if (name[cix] == '\0') {
        ich2 = 0;
      } else {
        ich2 = (int)name[cix] - 96;
      }

      N[ich1][ich2] += 1;
    }
  }

  return N;
}

class bigramNN {
public:
  torch::Tensor w1;
  torch::Tensor Z;
  torch::Tensor loss;

  bigramNN() {
    torch::Generator g =
        torch::make_generator<at::CPUGeneratorImpl>(2147483647);
    this->w1 = torch::randn({27, 27}, g, torch::requires_grad(true));
  }

  torch::Tensor forward(torch::Tensor X) {
    torch::Tensor xenc = torch::one_hot(X, 27).to(torch::kFloat32);
    // std::cout << "X:\n" << xenc << "\n";
    this->Z = torch::matmul(xenc, this->w1);
    // this->Z = torch::exp(this->Z);
    // this->Z = (this->Z) / this->Z.sum(1, true);
    return this->Z;
  }

  torch::Tensor backward(torch::Tensor Y) {
    (this->w1).retain_grad();
    /* Unoptimized itterative calculation of loss
     * this->loss = torch::tensor({0}).to(torch::kFloat32);
     * int ys = Y.size(0);
     *
     * std::cout << "Getting loss from " << ys << " outputs";
     * for (int ix = 0; ix < ys; ix++) {
     * this->loss += -(Z[ix][Y[ix].item()].log());
     * std::cout << "Loss " << ix << ": " << this->loss << "\n";
     * }
     */

    std::cout << "  Calculating loss...\n";
    // Calculates softmax and NLL loss in one optimized step
    this->loss = torch::nn::functional::cross_entropy(this->Z, Y);
    std::cout << "  Backward pass...\n";
    (this->loss).backward();
    return this->loss;
  }
};

int main() {
  // Get names from ../names.txt
  std::ifstream file("../names.txt");
  std::vector<std::string> names;
  int nc = get_names(&file, &names);

  std::vector<int> x;
  std::vector<int> y;

  std::cout << "Splicing " << nc << " names\n";
  // Get character inputs/expected outputs for first 20 words
  for (int nix = 0; nix < nc; nix++) {
    std::string name = names[nix];
    for (int cix = 0; cix < (name.size() + 1); cix++) {
      int ich1;
      int ich2;

      if (cix == 0) {
        ich1 = 0;
      } else {
        ich1 = (int)name[cix - 1] - 96;
      }

      if (name[cix] == '\0') {
        ich2 = 0;
      } else {
        ich2 = (int)name[cix] - 96;
      }

      x.push_back(ich1);
      y.push_back(ich2);
    }
  }

  torch::Tensor X = torch::tensor(x);
  torch::Tensor Y = torch::tensor(y);

  // std::cout << "X:\n" << X << "\n";
  // std::cout << "Y:\n" << Y << "\n";

  bigramNN simpleNN;
  std::cout << "Initial Weights:\n" << simpleNN.w1 << "\n";

  for (int i = 0; i < 500; i++) {
    std::cout << "Forward pass " << i << "...";
    simpleNN.forward(X);
    // std::cout << "Logits:\n" << simpleNN.Z << "\n";
    std::cout << " Complete.\n";

    std::cout << "Backward pass " << i << "...\n";
    simpleNN.backward(Y);
    std::cout << " Complete\n";
    std::cout << "Loss: " << simpleNN.loss << "\n";

    simpleNN.w1 = simpleNN.w1 - (simpleNN.w1.grad() * 25);
    // simpleNN.w1.retain_grad();
    // simpleNN.w1.grad().zero_();
  }

  /* BIGRAM
   * bigram character counts
   * torch::Tensor N = get_bigram_counts(names, nc);
   */

  std::cout << "Generating names...\n";
  torch::Generator g = torch::make_generator<at::CPUGeneratorImpl>(2147483647);
  std::vector<std::string> gen_names = {};
  // torch::Tensor P = N / N.sum(1, true); // char. prob dist. base on previous
  for (int n = 0; n < 20; n++) {
    std::string name("");
    int ch_ix = 0;
    char ch;
    do {
      torch::Tensor P = simpleNN.forward(torch::tensor({ch_ix})).softmax(1);
      ch_ix = torch::multinomial(P, 1, true, g).item<int>(); // char index
      ch = (char)(ch_ix + 96); // char (convert w/ asciiencoding)
      name = name.append(1, ch);
    } while (ch_ix != 0);
    std::cout << name << "\n";
    gen_names.push_back(name);
  }

  return 0;
}
