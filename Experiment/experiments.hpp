//#define DEBUG_LOG_ 1

#include <bayesian/evaluation/mdl.hpp>
#include <bayesian/learning/brute_force.hpp>
#include <bayesian/learning/greedy.hpp>
#include <bayesian/learning/simulated_annealing.hpp>
#include <bayesian/learning/stepwise_structure.hpp>
#include <bayesian/learning/stepwise_structure_hc.hpp>

std::size_t const iteration_num = 10; // 10

using EvaluationAlgorithm = bn::evaluation::mdl;

struct algorithm_holder {
    std::string name;
    std::function<double(bn::graph_t& graph, bn::sampler const& sampler)> function;
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

struct rms_60_method {
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
        auto const similar = std::sqrt(0.6 * std::pow(primary_similarity, 2) + 0.4* std::pow(secondary_similarity, 2));
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

struct rms_40_method {
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
        auto const similar = std::sqrt(0.4 * std::pow(primary_similarity, 2) + 0.6 * std::pow(secondary_similarity, 2));
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

struct average_60_method {
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
        auto const similar = 0.6 * primary_similarity + 0.4 * secondary_similarity;
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

struct average_40_method {
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
        auto const similar = 0.4 * primary_similarity + 0.6 * secondary_similarity;
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
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::previous_method> sshc(sampler);
            return sshc(graph, 0.0);
        }
    },
    {
        "sshc_previous_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::previous_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_previous_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::previous_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_previous_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::previous_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_same_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::same_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_same_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::same_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_same_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::same_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_rms60_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_60_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_rms60_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_60_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_rms60_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_60_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_rms50_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_50_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_rms50_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_50_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_rms50_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_50_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_rms40_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_40_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_rms40_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_40_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_rms40_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::rms_40_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_ave60_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_60_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_ave60_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_60_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_ave60_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_60_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_ave50_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_50_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_ave50_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_50_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_ave50_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_50_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    },
    {
        "sshc_ave40_10",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_40_method> sshc(sampler);
            return sshc(graph, 0.1);
        }
    },
    {
        "sshc_ave40_20",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_40_method> sshc(sampler);
            return sshc(graph, 0.2);
        }
    },
    {
        "sshc_ave40_30",
        [](bn::graph_t& graph, bn::sampler const& sampler)
        {
            bn::learning::stepwise_structure_hc<EvaluationAlgorithm, bn::learning::greedy, pruning_probability::average_40_method> sshc(sampler);
            return sshc(graph, 0.3);
        }
    }
};
