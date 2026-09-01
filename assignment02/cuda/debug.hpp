#include <fstream>

template <class Ele, class EleTrans, class Container, bool transform>
void print_matrix_to_file(const char *file, Container matrix, int m, int n,
                          EleTrans (*fn)(Ele) = nullptr) {
  std::ofstream of(file);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      auto idx = i * n + j;
      if constexpr (transform) {
        of << fn(matrix[idx]);
      } else {
        of << matrix[idx];
      }
      of << " ";
    }
    of << std::endl;
  }
}