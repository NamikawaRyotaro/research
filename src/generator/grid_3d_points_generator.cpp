#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
using namespace std;
int main()
{
  double xmin, xmax, ymin, ymax, zmin, zmax, split;
  std::cout << "xmin:";
  std::cin >> xmin;
  std::cout << "xmax:";
  std::cin >> xmax;
  std::cout << "ymin:";
  std::cin >> ymin;
  std::cout << "ymax:";
  std::cin >> ymax;
  std::cout << "zmin:";
  std::cin >> zmin;
  std::cout << "zmax:";
  std::cin >> zmax;
  std::cout << "split:";
  std::cin >> split;

  stringstream filename;
  filename << "camera_pos(x:[" << xmin << "," << xmax << ":" << split << "],";
  filename << "y:[" << ymin << "," << ymax << ":" << split << "],";
  filename << "z:[" << zmin << "," << zmax << ":" << split << "])";

  ofstream of("./resource/" + filename.str(), ios::trunc);

  for (int x = 0; x <= split; x++)
  {
    if (x != 0 && abs(xmax - xmin) <= numeric_limits<double>::epsilon())
      break;
    for (int y = 0; y <= split; y++)
    {
      if (y != 0 && abs(ymax - ymin) <= numeric_limits<double>::epsilon())
        break;
      for (int z = 0; z <= split; z++)
      {
        if (z != 0 && abs(zmax - zmin) <= numeric_limits<double>::epsilon())
          break;
        of << xmin + x * (xmax - xmin) / split << " ";
        of << ymin + y * (ymax - ymin) / split << " ";
        of << zmin + z * (zmax - zmin) / split << std::endl;
      }
    }
  }
  return 0;
}