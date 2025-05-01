#include "llama-hparams.h"

#include "ggml.h"

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head/n_head_kv;
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k * n_head_kv;
}

uint32_t llama_hparams::n_embd_k_s(uint32_t il) const {
    if (wkv_head_size != 0) {
        // for RWKV models
        return token_shift_count * n_embd;
    }

    // For Falcon Mamba2, use the correct intermediate size
    uint32_t intermediate_size = ssm_mamba_d_ssm > 0 ? ssm_mamba_d_ssm : ssm_d_inner;
    
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    return (ssm_d_conv > 0 ? ssm_d_conv : 0) * (intermediate_size + 2*ssm_n_group*ssm_d_state);
}

uint32_t llama_hparams::n_embd_v_s(uint32_t il) const {
    if (!recurrent_layer(il)) {
        return 0;
    }

    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    // For Falcon Mamba2, account for the head dimension structure
    if (ssm_head_dim > 0) {
        return ssm_d_state * ssm_dt_rank * ssm_head_dim;
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}

bool llama_hparams::recurrent_layer(uint32_t il) const {
    if (il < n_layer) {
        return recurrent_layer_arr[il];
    }
    GGML_ABORT("fatal error");
}
