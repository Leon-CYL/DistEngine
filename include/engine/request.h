#pragma once
#include <string>
#include <vector>
#include <cstddef>

namespace distengine
{
    enum class RequestState
    {
        Waiting,
        Running,
        Finished
    };

    enum class FinishReason {
        None,
        EOS,
        Length
    };

    class Request
    {
    private:
        std::string request_id_;
        std::vector<int> prompt_tokens_;
        std::vector<int> output_tokens_;
        std::size_t max_output_tokens_;
        int EOS; // EOS Token ID
        RequestState state_;
        FinishReason reason_;

    public:
        // Owns the supplied ID and prompt tokens. Starts Waiting with empty
        // output tokens and finish reason None. Throws std::invalid_argument
        // if the prompt is empty or max_output_tokens is zero.
        Request(std::string request_id, std::vector<int> prompt_tokens, std::size_t max_output_tokens, int EOS);

        // Changes Waiting to Running. Throws std::logic_error in any other
        // state, leaving the request unchanged.
        void start();

        // Appends one output token while Running, including EOS in the count.
        // Finishes with reason EOS if the token is EOS; otherwise finishes with
        // reason Length when the output budget is reached. EOS takes precedence
        // when both conditions hold. Otherwise remains Running.
        // Throws std::logic_error while Waiting or Finished, leaving the
        // request unchanged.
        void append_token(int token_id);
        bool is_finished() const;

        const std::string& id() const;
        const std::vector<int>& prompt_tokens() const;
        const std::vector<int>& output_tokens() const;
        std::size_t max_output_tokens() const;
        int eos_token_id() const;
        RequestState state() const;
        FinishReason finish_reason() const;
        std::size_t num_prompt_tokens() const;
        std::size_t num_output_tokens() const;
        std::size_t num_total_tokens() const;
    };
}
