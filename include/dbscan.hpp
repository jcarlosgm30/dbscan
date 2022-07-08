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

#pragma once


#include <nanoflann.hpp>


/*
 * DBSCAN implementation
 */
template <typename KDtreeType, typename ElementType>
void dbscan(const KDtreeType& tree, const std::shared_ptr<std::vector<ElementType>>& data,
            double epsilon, size_t min_pts, const nanoflann::SearchParams& params,
            std::vector<std::vector<size_t>>& clusters)
{
  std::vector<bool> visited;
  visited.reserve(data->size());
  std::vector<std::pair<uint, double>> neighbor_pts;
  std::vector<std::pair<uint, double>> neighbor_sub_pts;

  for (size_t i = 0; i < data->size(); i++)
  {
    // check if the point is visited
    if (visited[i])
    {
      continue;
    }
    visited[i] = true;

    // radius search around the unvisited point
    tree.radiusSearch(data->at(i).data(), epsilon, neighbor_pts, params);

    // check if the point is noise
    if (neighbor_pts.size() < static_cast<size_t>(min_pts))
    {
      continue;
    }

    // expand the clusters
    std::vector<size_t> cluster = std::vector<size_t>({ i });

    while (!neighbor_pts.empty())
    {
      const unsigned long nb_idx = neighbor_pts.back().first;
      neighbor_pts.pop_back();
      if (visited[nb_idx])
      {
        continue;
      }
      visited[nb_idx] = true;

      tree.radiusSearch(data->at(nb_idx).data(), epsilon, neighbor_sub_pts, params);

      if (neighbor_sub_pts.size() >= static_cast<size_t>(min_pts))
      {
        std::copy(neighbor_sub_pts.begin(), neighbor_sub_pts.end(),
                  std::back_inserter(neighbor_pts));
      }
      cluster.emplace_back(nb_idx);
    }
    clusters.emplace_back(std::move(cluster));
  }
}
