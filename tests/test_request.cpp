#include "engine/request.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using distengine::FinishReason;
using distengine::Request;
using distengine::RequestState;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Operation>
void expect_throw(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

// Copy the request so every public observation can be compared after rejection.
template <typename Operation>
void expect_rejected(Request& request, Operation operation) {
    const Request before = request;
    expect_throw<std::logic_error>(operation, "Expected invalid operation to throw");
    check(request.id() == before.id()
              && request.prompt_tokens() == before.prompt_tokens()
              && request.output_tokens() == before.output_tokens()
              && request.max_output_tokens() == before.max_output_tokens()
              && request.eos_token_id() == before.eos_token_id()
              && request.state() == before.state()
              && request.finish_reason() == before.finish_reason()
              && request.is_finished() == before.is_finished()
              && request.num_prompt_tokens() == before.num_prompt_tokens()
              && request.num_output_tokens() == before.num_output_tokens()
              && request.num_total_tokens() == before.num_total_tokens(),
          "Rejected operation changed the request");
}

void test_construction() {
    std::string id = "request-1";
    std::vector<int> prompt = {11, 12, 13};
    const Request request(id, prompt, 2, 99);
    id.clear();
    prompt.clear();
    check(request.id() == "request-1", "Request must own its ID");
    check(request.prompt_tokens() == std::vector<int>({11, 12, 13}),
          "Request must own its prompt");
    check(request.max_output_tokens() == 2 && request.eos_token_id() == 99,
          "Configuration was not preserved");
    check(request.state() == RequestState::Waiting && !request.is_finished(),
          "New request must be waiting");
    check(request.finish_reason() == FinishReason::None, "New request has a finish reason");
    check(request.output_tokens().empty() && request.num_output_tokens() == 0,
          "New request must have no output");
    check(request.num_prompt_tokens() == 3 && request.num_total_tokens() == 3,
          "Initial token counts are incorrect");
}

void test_validation() {
    expect_throw<std::invalid_argument>([] { Request request("empty", {}, 2, 99); },
                                        "Empty prompt must be rejected");
    expect_throw<std::invalid_argument>([] { Request request("zero", {11}, 0, 99); },
                                        "Zero output budget must be rejected");
}

void test_length_and_transitions() {
    Request request("length", {11, 12, 13}, 2, 99);
    expect_rejected(request, [&] { request.append_token(7); });
    request.start();
    check(request.state() == RequestState::Running && !request.is_finished(),
          "start() must enter Running");
    check(request.finish_reason() == FinishReason::None && request.output_tokens().empty(),
          "start() changed output or finish reason");
    expect_rejected(request, [&] { request.start(); });
    request.append_token(7);
    check(request.state() == RequestState::Running
              && request.finish_reason() == FinishReason::None,
          "Request finished before its output budget");
    check(request.num_prompt_tokens() == 3 && request.num_output_tokens() == 1
              && request.num_total_tokens() == 4,
          "Counts after one output token must be 3, 1, and 4");
    request.append_token(8);
    check(request.is_finished() && request.finish_reason() == FinishReason::Length,
          "Output budget must finish with reason Length");
    check(request.output_tokens() == std::vector<int>({7, 8})
              && request.num_output_tokens() == 2 && request.num_total_tokens() == 5,
          "Length completion has incorrect tokens or counts");
    expect_rejected(request, [&] { request.append_token(9); });
    expect_rejected(request, [&] { request.start(); });
}

void test_eos() {
    // Exercise early EOS and EOS exactly at the output limit.
    for (std::size_t budget : {3, 1}) {
        Request request("eos", {11}, budget, 99);
        request.start();
        request.append_token(99);
        check(request.is_finished() && request.finish_reason() == FinishReason::EOS,
              "EOS must take precedence over the length limit");
        check(request.output_tokens() == std::vector<int>({99})
                  && request.num_output_tokens() == 1 && request.num_total_tokens() == 2,
              "EOS must be stored and counted");
        expect_rejected(request, [&] { request.append_token(7); });
        expect_rejected(request, [&] { request.start(); });
    }
}

void test_prompt_eos() {
    Request request("prompt-eos", {99, 11}, 2, 99);
    check(request.state() == RequestState::Waiting && !request.is_finished()
              && request.finish_reason() == FinishReason::None,
          "EOS in the prompt must not finish the request");
    request.start();
    request.append_token(7);
    check(request.state() == RequestState::Running
              && request.finish_reason() == FinishReason::None,
          "Prompt EOS must not affect generation");
}

} // namespace

int main() {
    try {
        test_construction();
        test_validation();
        test_length_and_transitions();
        test_eos();
        test_prompt_eos();
    } catch (const std::exception& error) {
        std::cerr << "Request test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
