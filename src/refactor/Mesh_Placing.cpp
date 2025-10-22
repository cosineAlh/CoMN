/**
 * @file Mesh_Placing.cpp
 * @author booniebears
 * @brief
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <iostream>
#include <math.h>
#include <set>
#include <sstream>
#include <unordered_set>

#include "Mesh_Placing.h"
#include "defines.h"
#include "json.hpp"

namespace Refactor {
using json = nlohmann::json;

Mesh_Placing::Mesh_Placing(int filter, int bandWidth, int totalTiles, double Mesh_latency) : filter(filter), bandWidth(bandWidth), totalTiles(totalTiles), per_router_latency(Mesh_latency) {
}

/**
 * @brief task mapping using nearliest mapping algorithm.
 *
 */
MeshInfo Mesh_Placing::Mesh_mapping_energy_pipeline(map<pair<int, int>, int> traffic_mp) {
  vector<int> mapped_task;
  vector<vector<int>> located_router;
  vector<TileLocation> tileLocation; // virtual tile id -> location on chip (x,y,z)
  tileLocation.resize(totalTiles);
  NMAP(traffic_mp, tileLocation);
  ifstream f_traffic(PATH_TRAFFIC);
  int src, dst, packet;
  vector<vector<int>> traffic;
  while (f_traffic >> src >> dst >> packet) {
    traffic.push_back({src, dst, packet});
  }
  auto meshInfo = NMAP_pipeline(traffic, tileLocation);
  // For 3D, read number of tiers from SpecParam.json; default to 1.
  int tiers = 1;
  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  tiers = MAX(1, (int)specParam["NumTiers"].get<int>());

  int tiles_per_tier = (tiers > 0) ? (int)ceil((double)totalTiles / tiers) : totalTiles;
  int tile_rows = (int)ceil(sqrt(tiles_per_tier));
  int tile_cols = (int)ceil(sqrt(tiles_per_tier));
  ofstream f_place(PATH_PLACING, ios::app);
  f_place << "Total Rows: " << tile_rows << endl;
  f_place << "Total Columns: " << tile_cols << endl;
  if (tiers > 1) {
    f_place << "Total Tiers: " << tiers << endl;
  }

  // Build a mapping from tile id -> layer id by parsing meshconnect.txt
  vector<int> tileToLayer(totalTiles, -1);
  {
    ifstream f_mesh(PATH_MESHCONNECT);
    string line;
    while (getline(f_mesh, line)) {
      if (line.empty()) continue;
      istringstream iss(line);
      string tok;
      int prelayer = -1, nextlayer = -1;
      int pl_s = -1, pl_e = -1, nl_s = -1, nl_e = -1;
      while (iss >> tok) {
        if (tok == "prelayer:") {
          iss >> prelayer;
        } else if (tok == "nextlayer:") {
          iss >> nextlayer;
        } else if (tok == "prelayer_tile:") {
          iss >> pl_s >> pl_e;
        } else if (tok == "nextlayer_tile:") {
          iss >> nl_s >> nl_e;
        }
      }
      if (prelayer >= 0 && pl_s >= 0 && pl_e >= pl_s) {
        for (int t = pl_s; t <= pl_e && t < totalTiles; ++t) tileToLayer[t] = prelayer;
      }
      if (nextlayer >= 0 && nl_s >= 0 && nl_e >= nl_s) {
        for (int t = nl_s; t <= nl_e && t < totalTiles; ++t) tileToLayer[t] = nextlayer;
      }
    }
  }

  // Print a Grid for Physical Tile Location Mapping.
  for (int i = 0; i < totalTiles; i++) {
    int layer_id = (i >= 0 && i < (int)tileToLayer.size()) ? tileToLayer[i] : -1;
    if (tiers > 1) {
      f_place << "tile: " << i << " location: [" << tileLocation[i].x << ", " << tileLocation[i].y << ", " << tileLocation[i].z << "]" << " layer: " << layer_id << endl;
    } else {
      f_place << "tile: " << i << " location: [" << tileLocation[i].x << ", " << tileLocation[i].y << "]" << " layer: " << layer_id << endl;
    }
  }
  return meshInfo;
}

MeshInfo Mesh_Placing::NMAP_pipeline(vector<vector<int>> ip_index, vector<TileLocation> tileLocation) {
  // link_length: The number of store-and-forward times of Mesh routers;
  // Mesh_latency: total on-chip latency after scheduling the routing traffic;
  double link_length = 0, Mesh_latency = 0;
  // Read tiers here as well for scheduling decisions
  int tiers = 1;
  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  tiers = MAX(1, (int)specParam["NumTiers"].get<int>());

  vector<vector<int>> unscheduled_traffic = ip_index;
  set<int> mapped_tasks;
  for (int i = 0; i < totalTiles; i++) {
    mapped_tasks.insert(i);
  }

  set<int> unscheduled_tasks = mapped_tasks;
  while (!unscheduled_traffic.empty()) {
    // The tile to be removed from unscheduled_tasks. Also the destination tile
    // of next scheduling traffic.
    int target_task = 0;
    /*** 1. find the virtual tile id with greatest to-destination delay in the
     *** rest of traffic that has not been scheduled.***/
    // Build candidates directly from current traffic destinations to ensure progress
    std::unordered_set<int> candidate_dsts;
    for (const auto &vec : unscheduled_traffic) {
      candidate_dsts.insert(vec[1]);
    }
    double max_latency = -1;
    for (int task : candidate_dsts) {
      double routing_latency = 0;
      for (const auto &vec : unscheduled_traffic) {
        int src = vec[0], dst = vec[1], packet = vec[2];
        if (task == dst) {
          int x_from = tileLocation[src].x, y_from = tileLocation[src].y, z_from = tileLocation[src].z;
          int x_to = tileLocation[task].x, y_to = tileLocation[task].y, z_to = tileLocation[task].z;
          // The num of routers used to transfer packets
          int routers = abs(x_to - x_from) + abs(y_to - y_from) + abs(z_to - z_from) + 1;
          routing_latency += (1.0 * packet / bandWidth + 1.0 * filter / bandWidth * routers) * per_router_latency;
        }
      }
      if (max_latency < routing_latency) {
        max_latency = routing_latency;
        target_task = task;
      }
    }

    /*** 2. Iteratively schedule traffic where "target_task" is the destination.
     *** After scheduling, traffic is removed from unscheduled_traffic. ***/
    unscheduled_tasks.erase(target_task);
    double latency;

    for (auto it = unscheduled_traffic.begin(); it != unscheduled_traffic.end(); it++) {
      auto vec = *it;
      int src = vec[0], dst = vec[1], packet = vec[2];
      if (target_task == dst) {
        int x_from = tileLocation[src].x, y_from = tileLocation[src].y, z_from = tileLocation[src].z;
        int x_to = tileLocation[target_task].x, y_to = tileLocation[target_task].y, z_to = tileLocation[target_task].z;

        unscheduled_traffic.erase(it);
        it--; // vector traverse method. do not work for map.

        int routers = abs(x_to - x_from) + abs(y_to - y_from) + abs(z_to - z_from) + 1;
        // For the whole packet to be sent onto link;
        double send_latency = 1.0 * packet / bandWidth;
        // For a router to transfer a filter;
        double transfer_latency = 1.0 * filter / bandWidth;
        if (tiers > 1) {
          latency = schedule_idle_3d(x_to, y_to, z_to, x_from, y_from, z_from, send_latency, transfer_latency);
        } else {
          latency = schedule_idle(x_to, y_to, x_from, y_from, send_latency, transfer_latency);
        }
        latency *= per_router_latency;
        // Figure out the longest latency among all the routing traffic.
        if (Mesh_latency < latency) {
          Mesh_latency = latency;
        }
        link_length += routers * packet;
      }
    }
  }
  cout << "link_length = " << link_length << ", Mesh_latency = " << Mesh_latency << endl;
  return MeshInfo{link_length, Mesh_latency};
}

/**
 * @brief Greedy Algorithm NMAP for Calculating Placing tiles physically.
 * TODO: Time costing!!!
 */
void Mesh_Placing::NMAP(map<pair<int, int>, int> traffic_mp, vector<TileLocation> &tileLocation) {
  // For a given virtual tile id, return the physical tile location.
  // Suppose the chip has infinite area, and tiles are expected to be allocated
  // physically in a "RECTANGULAR" manner.
  set<int> unmapped_tasks; // unmapped set of virtual tiles(id stored in it)
  set<int> mapped_tasks;
  set<TileLocation> unallocated_tiles; // tileLocations not allocated yet
  set<TileLocation> allocated_tiles;
  cout << "totalTiles = " << totalTiles << endl;
  for (int i = 0; i < totalTiles; i++) {
    unmapped_tasks.insert(i);
  }

  int tiers = 1;
  ifstream f_spec(PATH_SPEC_PARAM);
  json specParam = json::parse(f_spec);
  tiers = MAX(1, (int)specParam["NumTiers"].get<int>());

  int tiles_per_tier = (tiers > 0) ? (int)ceil((double)totalTiles / tiers) : totalTiles;
  int tile_rows = (int)ceil(sqrt(tiles_per_tier));
  int tile_cols = (int)ceil(sqrt(tiles_per_tier));
  cout << "tile_rows = " << tile_rows << endl;
  cout << "tile_cols = " << tile_cols << endl;
  if (tiers > 1) {
    cout << "tiers = " << tiers << endl;
  }
  for (int z = 0; z < MAX(1, tiers); ++z) {
    for (int i = 0; i < tile_rows; i++) {
      for (int j = 0; j < tile_cols; j++) {
        unallocated_tiles.insert({i, j, z});
      }
    }
  }

  /*** 1: Select tile with the greatest traffic, and place it in the center
   *** of the rectangle. ***/
  vector<int> packet_transmission(totalTiles, 0);
  for (auto &pr : traffic_mp) {
    auto key = pr.first;
    int src = key.first, dst = key.second, packet = pr.second;
    packet_transmission[src] += packet;
    packet_transmission[dst] += packet;
  }
  int max_traffic_tile = 0, max_traffic = -1;
  for (int i = 0; i < totalTiles; i++) {
    if (max_traffic < packet_transmission[i]) {
      max_traffic = packet_transmission[i];
      max_traffic_tile = i;
    }
  }

  unmapped_tasks.erase(max_traffic_tile);
  mapped_tasks.insert(max_traffic_tile);
  unallocated_tiles.erase({tile_rows / 2, tile_cols / 2, 0});
  allocated_tiles.insert({tile_rows / 2, tile_cols / 2, 0});
  tileLocation[max_traffic_tile] = {tile_rows / 2, tile_cols / 2, 0};

  /*** 2: Iteratively select tile with the next greatest traffic from tiles
   *** already mapped onto chip physically, and place it on
   *** the location with the minimum communication(∑distance * traffic)
   *** to the same tiles. ***/
  // int cnt = 0;
  while (!unmapped_tasks.empty()) {
    // cnt++;
    // cout << "NMAP Greedy Mapping Iteration: " << cnt << endl;
    int min_communication = INF;
    int max_traffic = 0;
    int max_unmapped_task = 0; // unmapped task with greatest traffic

    map<pair<int, int>, int> mp; // Traffic between unmapped and mapped task.
    map<pair<int, int>, int> tmp_mp;
    // Max Traffic with mapped_tasks
    for (int task : unmapped_tasks) {
      int traffic = 0;
      tmp_mp.clear();
      for (int mapped_task : mapped_tasks) {
        tmp_mp[{task, mapped_task}] += traffic_mp[{task, mapped_task}] + traffic_mp[{mapped_task, task}];
        traffic += traffic_mp[{task, mapped_task}] + traffic_mp[{mapped_task, task}];
      }
      if (traffic >= max_traffic) {
        mp = tmp_mp;
        max_traffic = traffic;
        max_unmapped_task = task;
      }
    }
    
    // Min Communication
    TileLocation bestLocation;
    for (auto loc : unallocated_tiles) {
      double communication = 0;
      for (int mapped_task : mapped_tasks) {
        double total_packet = 0;
        TileLocation mapped_loc = tileLocation[mapped_task];
        double manhatten_dis = abs(mapped_loc.x - loc.x) + abs(mapped_loc.y - loc.y) + abs(mapped_loc.z - loc.z);
        total_packet = mp[{max_unmapped_task, mapped_task}];
        communication += total_packet * manhatten_dis;
      }
      if (communication < min_communication) {
        min_communication = communication;
        bestLocation = loc;
      }
    }

    unmapped_tasks.erase(max_unmapped_task);
    mapped_tasks.insert(max_unmapped_task);
    unallocated_tiles.erase(bestLocation);
    allocated_tiles.insert(bestLocation);
    tileLocation[max_unmapped_task] = bestLocation;
  }

  cout << "NMAP Greedy Mapping Finished!!!" << endl;
}

/**
 * @brief Calculate the on-chip latency of the whole NoC. Traffic congestion in
 * routing are also considered in this method.
 *
 * @param transfer_latency The ideal routing latency of a "filter" between two
 * adjacent routers without considering any congestion. Also, this "latency" has
 * not been multiplied by per_router_latency.
 * @param send_latency The latency for a whole packet to be sent onto link.
 */
double Mesh_Placing::schedule_idle(int x_to, int y_to, int x_from, int y_from, double send_latency, double transfer_latency) {
  // The final On-chip latency returned after scheduling is supposed to be
  // greater than the "latency" without considering any congestion.

  // Simplified Assumption One: One router can only handle one forward request
  // at one time; send and receive functions are incompatible. Subsequent
  // forwarding requests need to wait for the completion of the previous
  // forwarding request.

  // Simplified Assumption Two: All routers employ XY routing, i.e., packets are
  // first transferred in the "row" direction, and then tranferred in the
  // "column" direction. 

  // Simplified Assumption Three: All routers have infinite storage that can
  // hold all the packets to be transferred;

  // If source equals destination, handle locally and return
  if (x_from == x_to && y_from == y_to) {
    timeTable[{x_from, y_from, 0}].start = timeTable[{x_from, y_from, 0}].end + transfer_latency;
    timeTable[{x_from, y_from, 0}].end = timeTable[{x_from, y_from, 0}].start + send_latency;
    return timeTable[{x_from, y_from, 0}].end;
  }

  // Move direction per axis: -1, 0, +1
  int x_inc = (x_from < x_to) ? 1 : ((x_from > x_to) ? -1 : 0);
  int y_inc = (y_from < y_to) ? 1 : ((y_from > y_to) ? -1 : 0);
  // Starting point:
  timeTable[{x_from, y_from, 0}].start = timeTable[{x_from, y_from, 0}].end + transfer_latency;
  timeTable[{x_from, y_from, 0}].end = timeTable[{x_from, y_from, 0}].start + send_latency;
  TileLocation nxtLoc = (y_from == y_to) ? TileLocation{x_from + x_inc, y_from, 0}
                                         : TileLocation{x_from, y_from + y_inc, 0};
  if (timeTable[{x_from, y_from, 0}].start < timeTable[nxtLoc].end) {
    // Congestion happened
    timeTable[{x_from, y_from, 0}].end = timeTable[nxtLoc].end + send_latency;
  }

  // For the rest points, use three "locations" to iteratively find the answer.
  TileLocation thisLoc = nxtLoc;
  TileLocation preLoc = TileLocation{x_from, y_from, 0};
  nxtLoc = (thisLoc == TileLocation{x_to, y_to, 0}) ? thisLoc
           : (thisLoc.y == y_to) ? TileLocation{thisLoc.x + x_inc, thisLoc.y, 0}
                                 : TileLocation{thisLoc.x, thisLoc.y + y_inc, 0};

  while (thisLoc != TileLocation{x_to, y_to, 0}) {
    if (timeTable[preLoc].start < timeTable[thisLoc].end) {
      // Congestion happened
      timeTable[thisLoc].start = timeTable[thisLoc].end + transfer_latency;
    } else {
      // No congestion
      timeTable[thisLoc].start = timeTable[preLoc].start + transfer_latency;
    }

    if (timeTable[thisLoc].start < timeTable[nxtLoc].end) {
      // Congestion happened
      timeTable[thisLoc].end = timeTable[nxtLoc].end + send_latency;
    } else {
      // No congestion
      timeTable[thisLoc].end = timeTable[thisLoc].start + send_latency;
    }
    preLoc = thisLoc;
    thisLoc = nxtLoc;
    nxtLoc = (thisLoc == TileLocation{x_to, y_to, 0}) ? thisLoc
      : (thisLoc.y == y_to) ? TileLocation{thisLoc.x + x_inc, thisLoc.y, 0}
             : TileLocation{thisLoc.x, thisLoc.y + y_inc, 0};
  }

  // Jump out of the loop, nxtLoc = thisLoc = TileLocation{x_to, y_to};
  if (timeTable[preLoc].start < timeTable[{x_to, y_to, 0}].end) {
    // Congestion happened
    timeTable[{x_to, y_to, 0}].start = timeTable[{x_to, y_to, 0}].end + transfer_latency;
  } else {
    // No congestion
    timeTable[{x_to, y_to, 0}].start = timeTable[preLoc].start + transfer_latency;
  }

  timeTable[{x_to, y_to, 0}].end = timeTable[{x_to, y_to, 0}].start + send_latency;

  return timeTable[{x_to, y_to, 0}].end;
}

// 3D scheduling: route along X -> Y -> Z, considering per-hop timing similar to 2D.
double Mesh_Placing::schedule_idle_3d(int x_to, int y_to, int z_to, int x_from, int y_from, int z_from, double send_latency, double transfer_latency) {
  // For now, we reuse the 2D timing model across segments and sum; a full 3D contention model could be built similarly per segment.
  // Segment 1: X/Y in source z
  double end_xy = schedule_idle(x_to, y_to, x_from, y_from, send_latency, transfer_latency);
  // Segment 2: Z hops from z_from to z_to, treated as linear chain in Z
  int z_inc = (z_from < z_to) ? 1 : -1;
  TileLocation preLoc = {x_to, y_to, z_from};
  for (int z = z_from; z != z_to; z += z_inc) {
    TileLocation thisLoc = {x_to, y_to, z};
    TileLocation nxtLoc = {x_to, y_to, z + z_inc};
    if (timeTable[preLoc].start < timeTable[thisLoc].end) {
      timeTable[thisLoc].start = timeTable[thisLoc].end + transfer_latency;
    } else {
      timeTable[thisLoc].start = timeTable[preLoc].start + transfer_latency;
    }
    if (timeTable[thisLoc].start < timeTable[nxtLoc].end) {
      timeTable[thisLoc].end = timeTable[nxtLoc].end + send_latency;
    } else {
      timeTable[thisLoc].end = timeTable[thisLoc].start + send_latency;
    }
    preLoc = thisLoc;
  }
  // Destination point at (x_to,y_to,z_to)
  if (timeTable[preLoc].start < timeTable[{x_to, y_to, z_to}].end) {
    timeTable[{x_to, y_to, z_to}].start = timeTable[{x_to, y_to, z_to}].end + transfer_latency;
  } else {
    timeTable[{x_to, y_to, z_to}].start = timeTable[preLoc].start + transfer_latency;
  }
  timeTable[{x_to, y_to, z_to}].end = timeTable[{x_to, y_to, z_to}].start + send_latency;
  return timeTable[{x_to, y_to, z_to}].end;
}

} // namespace Refactor