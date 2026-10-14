#pragma once

#define __mram_ptr

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <cstdint>
#include <string>

#include "task_utils.hpp"
#include "task_framework_host.hpp"
#include "compile.hpp"
#include "pim_interface_header.hpp"
#include "random_generator.hpp"

void init_dpus() {
    printf("\n********** INIT DPUS **********\n");
#ifdef DPU_INIT_ON
    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init();
    IO_Task_Batch* batch = io->alloc<dpu_init_task, empty_task_reply>(direct);
    parfor_wrap(0, nr_of_dpus, [&](size_t i) {
        auto it = (dpu_init_task*)batch->push_task_zero_copy(i, -1, false);
        it->dpu_id = (int32_t)i;
        it->nr_of_dpus = (int32_t)nr_of_dpus;
    });
    io->finish_task_batch();
    ASSERT(io->exec());
    io->reset();
#endif
    printf("\n****** INIT DPUS Finished ******\n");
}

void host_init(std::string interface_type = "upmem") {
    srand(0);
    rn_gen::init();
    init_wram_save_pos();
    init_io_managers();
    // Simulator: DPU_ALLOCATE_ALL yields a single 64-DPU rank, so request NR_DPUS explicitly.
    dpu_control::alloc(NR_DPUS);
    namespace_pim_interface::pim_interface_init(dpu_set, interface_type);
    namespace_pim_interface::do_not_free_dpu_set_when_delete();
    IO_Manager::using_upmem_interface = (interface_type != "direct");

    cpu_coverage_timer->start();
    dpu_binary_switch_to(dpu_binary::insert_binary);
    init_dpus();
    cpu_coverage_timer->end();
    cpu_coverage_timer->reset();
    pim_coverage_timer->reset();
    printf("---------------- Timer Init ---------------\n");
}

void host_end() {
    namespace_pim_interface::pim_interface_delete();
    dpu_control::free_the_dpus();
}

void sample_points(parlay::sequence<vectorT>& wp, size_t n) {
    double interval = static_cast<double>(wp.size()) / n;
    wp = parlay::tabulate(n, [&](size_t i) {
        size_t idx = static_cast<size_t>(i * interval);
        if (idx >= wp.size()) idx = wp.size() - 1;
        return wp[idx];
    });
}