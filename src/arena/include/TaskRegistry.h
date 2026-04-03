#pragma once

#include "Policy.h"
#include <functional>
#include <unordered_map>

// Singleton factory registry that maps task names to Policy constructors.
//
// Tasks self-register via the REGISTER_TASK macro, which creates a static
// Registrar object whose constructor calls register_task() before main() runs.
// main() then looks up the desired task by name and calls create().
//
// Usage:
//   // In a task .cpp file:
//   REGISTER_TASK("t1-velocity-flat", T1VelocityFlat);
//
//   // In main():
//   auto policy = TaskRegistry::instance().create("t1-velocity-flat");
class TaskRegistry {
public:
    using Factory = std::function<std::unique_ptr<Policy>()>;

    static TaskRegistry& instance();

    void register_task(const std::string& name, Factory factory);

    // Create a fresh Policy for the given task name.
    // Throws std::runtime_error if the task is not registered.
    std::unique_ptr<Policy> create(const std::string& name) const;
    bool has(const std::string& name) const;

    // Nested helper: construct one of these as a static local variable
    // to register a task before main() runs.
    struct Registrar {
        Registrar(const std::string& name, Factory factory) {
            TaskRegistry::instance().register_task(name, std::move(factory));
        }
    };

private:
    TaskRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

// Convenience macro — mirrors REGISTER_MODULE from spqrbooster.
// Place once per task .cpp file (outside any namespace/class).
#define REGISTER_TASK(task_name, ClassName)                                 \
    static TaskRegistry::Registrar __##ClassName##_registrar(               \
        task_name,                                                          \
        []() -> std::unique_ptr<Policy> {                                   \
            return std::make_unique<ClassName>();                           \
        }                                                                   \
    );
