#include "engine/request.h"

#include <utility>
#include <stdexcept>

distengine::Request::Request(std::string request_id, std::vector<int> prompt_tokens, std::size_t max_output_tokens, int EOS)
    : request_id_(std::move(request_id)),
      prompt_tokens_(std::move(prompt_tokens)),
      max_output_tokens_(max_output_tokens),
      EOS(EOS),
      state_(distengine::RequestState::Waiting),
      reason_(distengine::FinishReason::None)
{
    if (prompt_tokens_.empty())
    {
        throw std::invalid_argument("Prompt Tokens are empty! Can't Generate Tokens without Given Prompt!\n");
    }

    if (max_output_tokens_ == 0)
    {
        throw std::invalid_argument("Maximum output tokens must be greater than zero.\n");
    }
}

void distengine::Request::start()
{
    if (this->state_ != distengine::RequestState::Waiting)
    {
        throw std::logic_error("Request must be waiting state when call start()!\n");
    }
    this->state_ = distengine::RequestState::Running;
}

void distengine::Request::append_token(int token_id)
{
    if (this->state_ != distengine::RequestState::Running)
    {
        throw std::logic_error("Request must be Running state to append Token!\n");
    }

    this->output_tokens_.push_back(token_id);

    if (token_id == this->EOS)
    {
        this->state_ = distengine::RequestState::Finished;
        this->reason_ = distengine::FinishReason::EOS;
    }
    else if (this->output_tokens_.size() == this->max_output_tokens_)
    {
        this->state_ = distengine::RequestState::Finished;
        this->reason_ = distengine::FinishReason::Length;
    }
}

bool distengine::Request::is_finished() const
{
    return this->state_ == distengine::RequestState::Finished;
}

const std::string &distengine::Request::id() const
{
    return this->request_id_;
}

const std::vector<int> &distengine::Request::prompt_tokens() const {
    return this->prompt_tokens_;
}

const std::vector<int> &distengine::Request::output_tokens() const {
    return this->output_tokens_;
}

std::size_t distengine::Request::max_output_tokens() const {
    return this->max_output_tokens_;
}

int distengine::Request::eos_token_id() const {
    return this->EOS;
}

distengine::RequestState distengine::Request::state() const {
    return this->state_;
}

distengine::FinishReason distengine::Request::finish_reason() const {
    return this->reason_;
}

std::size_t distengine::Request::num_prompt_tokens() const {
    return this->prompt_tokens_.size();
}


std::size_t distengine::Request::num_output_tokens() const {
    return this->output_tokens_.size();
}

std::size_t distengine::Request::num_total_tokens() const {
    return this->num_prompt_tokens() + this->num_output_tokens();
}
