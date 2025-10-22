/**
 * @file Perf_Evaluator.h
 * @author booniebears
 * @brief Evaluate the performance info of the whole CIM arch.
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PERF_EVALUATOR_H_
#define PERF_EVALUATOR_H_

namespace Refactor {

// The interface of Calculating the Performance of all modules required.
void PPA_cost();

void HISIM();

// Performance of PE
double PE_Perf();

// Performance of Orion(routers)
double NOC_Perf(int inPorts, int outPorts, int v_channels, double freq, int featureSize);

} // namespace Refactor

#endif // !PERF_EVALUATOR_H_
