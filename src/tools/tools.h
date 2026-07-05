#ifndef TOOLS_H
#define TOOLS_H

#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

// file -> ifstream to the file
//
inline int get_items(std::ifstream &file, std::vector<std::string> &items) {
  std::string line;
  while (std::getline(file, line)) {
    items.push_back(line);
  }

  return 0;
}

// names -> string vector of names
// context_len -> context len for x
// x, y -> empty vectors to append x and y
inline int get_xy(std::vector<std::string> &names, int context_len,
                  std::vector<int> &x, std::vector<int> &y) {
  for (int nix = 0; nix < names.size(); nix++) {
    std::string name = names[nix];
    for (int cix = 0; cix < (name.size() + 1); cix++) {
      // X
      int chx;
      for (int cx = (context_len - 1); (cx + 1) > 0; cx--) {
        if (cix - cx > 0) {
          chx = (int)name[cix - cx];
        } else {
          chx = 0;
        }
        x.push_back(chx);
      }

      int chy;
      if (name[cix] == '\0') {
        chy = 0;
      } else {
        chy = (int)name[cix];
      }

      y.push_back(chy);
    }
  }

  return 0;
}

inline torch::Tensor get_bigram_counts(std::string *names, int nc) {
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

#endif
