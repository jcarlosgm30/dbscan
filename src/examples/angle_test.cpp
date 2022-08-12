/**
 * Copyright (c) 2022 Jose Carlos Garcia (jcarlos3094@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author Jose Carlos Garcia
 *
 */

#include <iostream>
#include <memory>
#include <random>
#include "dbscan.hpp"

template <typename T>
T shortestAngularDist(T from, T to)
{
  const T angle = to - from;
  T angle_norm = fmod(fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
  if (angle_norm > M_PI)
  {
    angle_norm -= 2.0 * M_PI;
  }
  return std::abs(angle_norm);
}

template <class T, class DataSource, typename _DistanceType = T, typename AccessorType = uint32_t>
struct Shortest_Angle_Adaptor
{
  using ElementType = T;
  using DistanceType = _DistanceType;

  const DataSource& data_source;

  Shortest_Angle_Adaptor(const DataSource& _data_source) : data_source(_data_source)
  {
  }

  inline DistanceType evalMetric(const T* a, const AccessorType b_idx, size_t size) const
  {
    DistanceType result = DistanceType();
    for (size_t i = 0; i < size; ++i)
    {
      result = shortestAngularDist<ElementType>(a[i], data_source.kdtree_get_pt(b_idx, i));
    }
    return result;
  }

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const
  {
    return shortestAngularDist<ElementType>(a, b);
  }
};

template <typename T>
struct KdtreeAngleAdaptor
{
  KdtreeAngleAdaptor(const std::vector<T>& angle) : angle_(angle)
  {
  }
  const std::vector<T>& angle_;

  inline size_t kdtree_get_point_count() const
  {
    return angle_.size();
  }
  inline double kdtree_get_pt(const size_t idx, const size_t dim = 0) const
  {
    return angle_[idx].at(0);
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const
  {
    return false;
  }
};

int main()
{
  // define data
  const int dim = 1;
  std::shared_ptr<std::vector<std::array<double, dim>>> data =
      std::make_shared<std::vector<std::array<double, dim>>>();

  // generate artificial data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distr(-M_PI, M_PI);
  size_t N = 100;
  size_t n = 0;

  data->reserve(N);
  while (n < N)
  {
    data->push_back({ { distr(gen) } });
    n++;
  }

  // build kdtree:
  auto adapt = KdtreeAngleAdaptor<std::array<double, dim>>(*data);

  using my_kd_tree_t =
      nanoflann::KDTreeSingleIndexAdaptor<Shortest_Angle_Adaptor<double, decltype(adapt)>, decltype(adapt), dim /* dim */
                                          >;

  my_kd_tree_t index(dim /*dim*/, adapt, { 10 /* max leaf */ });

  index.buildIndex();

  std::vector<std::vector<size_t>> clusters;
  const double epsilon = 0.0872665;
  const int min_pts = 1;
  dbscan<my_kd_tree_t, std::array<double, dim>>(index, data, epsilon, min_pts,
                                                nanoflann::SearchParams(10), clusters);

  // RESULTS
  for (const auto& cluster : clusters)
  {
    std::cout << "Cluster: " << std::endl;
    for (const auto& id : cluster)
    {
      std::cout << "- Data " << id << " : " << *data->at(id).data() << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
