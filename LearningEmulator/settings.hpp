#define DEBUG_LOG_ 1

#include <bayesian/evaluation/mdl.hpp>
#include <bayesian/learning/brute_force.hpp>
#include <bayesian/learning/greedy.hpp>
#include <bayesian/learning/simulated_annealing.hpp>
#include <bayesian/learning/stepwise_structure.hpp>
#include <bayesian/learning/stepwise_structure_hc.hpp>

using EvaluationAlgorithm = bn::evaluation::mdl;

struct algorithm_holder {
    std::string name;
    std::function<std::vector<combine_phase>(bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)> function;
};

namespace pruning_probability {

struct previous_method {
    double p1(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return std::pow(alpha, outer_similarity / average_similar);
    }
    double p2(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return std::pow(alpha, primary_similarity / inner_similarity);
    }
    double p3(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return 1.0;
    }
};

struct same_method {
    double p1(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return std::pow(alpha, outer_similarity / average_similar);
    }

    double p2(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return std::pow(alpha, outer_similarity / average_similar);
    }

    double p3(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return 1.0;
    }
};

struct rms_50_method {
    double p1(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return std::pow(alpha, outer_similarity / average_similar);
    }

    double p2(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        auto const similar = std::sqrt(0.5 * std::pow(primary_similarity, 2) + 0.5 * std::pow(secondary_similarity, 2));
        return std::pow(alpha, similar / inner_similarity);
    }

    double p3(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return 1.0;
    }
};

struct average_50_method {
    double p1(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return std::pow(alpha, outer_similarity / average_similar);
    }

    double p2(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        auto const similar = 0.5 * primary_similarity + 0.5 * secondary_similarity;
        return std::pow(alpha, similar / inner_similarity);
    }

    double p3(
        double const alpha, double const average_similar,
        double const inner_similarity, double const outer_similarity,
        double const primary_similarity, double const secondary_similarity
        ) const
    {
        return 1.0;
    }
};
}

std::vector<algorithm_holder> const algorithms = {
    {
        "sshc_00",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::previous_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.0);
            return sshc.record();
        }
    },
    {
        "sshc_10",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::previous_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.1);
            return sshc.record();
        }
    },
    {
        "sshc_20",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::previous_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.2);
            return sshc.record();
        }
    },
    {
        "sshc_30",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::previous_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.3);
            return sshc.record();
        }
    },
    {
        "sshc_same_10",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::same_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.1);
            return sshc.record();
        }
    },
    {
        "sshc_same_20",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::same_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.2);
            return sshc.record();
        }
    },
    {
        "sshc_same_30",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::same_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.3);
            return sshc.record();
        }
    },
    {
        "sshc_rms50_10",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::rms_50_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.1);
            return sshc.record();
        }
    },
    {
        "sshc_rms50_20",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::rms_50_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.2);
            return sshc.record();
        }
    },
    {
        "sshc_rms50_30",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::rms_50_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.3);
            return sshc.record();
        }
    },
    {
        "sshc_ave50_10",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::average_50_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.1);
            return sshc.record();
        }
    },
    {
        "sshc_ave50_20",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::average_50_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.2);
            return sshc.record();
        }
    },
    {
        "sshc_ave50_30",
        [](bn::graph_t& graph, bn::graph_t const& teacher_graph, bn::sampler const& sampler, logfile_t::learning_unit const& log)
        {
            sshc_emulator<EvaluationAlgorithm, from_teacher, pruning_probability::average_50_method> sshc(teacher_graph, sampler, log);
            sshc(graph, 0.3);
            return sshc.record();
        }
    }
};
