# -*- coding: utf-8 -*-
"""
W7 第 ⑥ 站：更新 —— 训练循环。

前五站都是「算一个数」，这一站是唯一会**改变模型**的一站。
核心只有 5 行：

    logits = model(xb)          # 前进：算出预测
    loss   = lm_loss(...)       # 对答案：得到一个数
    opt.zero_grad()             # 清空上一次的梯度
    loss.backward()             # 反向：算出每个参数的梯度
    opt.step()                  # 更新：按梯度挪一小步

这 5 行重复 STEPS 次，loss 就会往下掉。

新 API 只有 4 个：torch.optim.AdamW / .backward() / .step() / .zero_grad()

跑：
  & "D:\\anaconda3\\envs\\aigc\\python.exe" -X utf8 src\\w7_train.py

只留 2 个 TODO。
"""
import math
import time

import torch

from w7_corpus import BATCH_SIZE, BLOCK_SIZE, build_corpus, build_vocab, encode, get_batch, split_train_val
from w7_gpt import GPT
from w7_model import Config, lm_loss

STEPS = 600          # 训练多少步
EVAL_EVERY = 100     # 每多少步体检一次
LR = 1e-3            # 学习率
SEED = 1337


# ---------------------------------------------------------------- 体检
@torch.no_grad()
def estimate_loss(model, train, val, iters=10):
    """在固定的 10 个 batch 上算平均 loss。

    为什么要固定？因为训练时的 batch 是随机抽的，直接用随机 batch 比较，
    两次之间的差别可能只是「抽到的样本难易不同」，看不出真实趋势。
    固定同一批样本，变化才全部来自模型本身。
    """
    model.eval()
    saved = torch.get_rng_state()
    torch.manual_seed(2024)                     # 固定抽样，保证每次比的是同一批
    out = {}
    for name, data in (("train", train), ("val", val)):
        losses = torch.zeros(iters)
        for i in range(iters):
            xb, yb = get_batch(data)
            losses[i] = lm_loss(model(xb), yb).item()
        out[name] = losses.mean().item()
    torch.set_rng_state(saved)                  # 还原，不打扰训练自己的随机性
    model.train()
    return out


def main():
    torch.manual_seed(SEED)

    text = build_corpus(verbose=False)
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    train, val = split_train_val(data)

    cfg = Config()
    cfg.vocab_size = len(chars)

    model = GPT(cfg)
    model.train()                               # 打开 dropout

    # TODO-1  造优化器。它需要两样东西：要更新的参数、学习率。
    #         提示：torch.optim.AdamW(model.parameters(), lr=LR)
    #         为什么用 model.parameters() 而不是自己列一个列表？
    #           因为只有它知道哪些张量被注册成「参数」—— 这正是 ③ 站
    #           必须用 nn.ModuleList 而不是普通 list 的原因。
    opt = torch.optim.AdamW(model.parameters(), lr=LR)   # <-- TODO-1

    if not isinstance(opt, torch.optim.Optimizer):
        raise SystemExit(
            "\n[TODO-1 未完成] 优化器没造出来。\n"
            "  写：opt = torch.optim.AdamW(model.parameters(), lr=LR)\n"
        )

    n_params = sum(p.numel() for p in model.parameters())
    if n_params != 181120:
        raise SystemExit(
            "\n[警告] 优化器收到的参数量是 %s，应该是 181,120。\n"
            "  少了说明有子模块没被注册（检查 ③ 站的 nn.ModuleList）。\n" % format(n_params, ",")
        )

    print("=" * 62)
    print("训练配置")
    print("=" * 62)
    print("  参数量    %s" % format(n_params, ","))
    print("  训练/验证  %d / %d 字符" % (len(train), len(val)))
    print("  步数      %d     学习率 %.0e     每步 %d×%d=%d 个字符位置"
          % (STEPS, LR, BATCH_SIZE, BLOCK_SIZE, BATCH_SIZE * BLOCK_SIZE))

    print()
    print("=" * 62)
    print("训练（每 %d 步体检一次）" % EVAL_EVERY)
    print("=" * 62)
    print("  %-6s %10s %10s %10s %8s" % ("步", "本步loss", "train均", "val均", "困惑度"))
    print("  " + "-" * 50)

    t0 = time.time()
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            ev = estimate_loss(model, train, val)
            print("  %-6d %10s %10.4f %10.4f %8.2f"
                  % (step, "-", ev["train"], ev["val"], math.exp(ev["val"])))

        xb, yb = get_batch(train)

        # ===== 核心 5 行 =====
        logits = model(xb)              # ① 前进：算出 (B,T,V) 的预测
        loss = lm_loss(logits, yb)      # ② 对答案：压成一个数

        # TODO-2  清空上一次留下的梯度。
        #         为什么必须清？因为 PyTorch 的 .backward() 是「累加」不是「覆盖」。
        #         不清的话，第 600 步的梯度里会混着前 599 步的残留。
        #         提示：opt.zero_grad()
        opt.zero_grad()   # <-- TODO-2

        loss.backward()                 # ③ 反向：每个参数拿到自己的梯度
        opt.step()                      # ④ 更新：按梯度挪一步
        # =====================

    print("  " + "-" * 50)
    print("  用时 %.1f 秒" % (time.time() - t0))

    final = estimate_loss(model, train, val, iters=20)

    # 结果守卫：有些错代码能跑、不报错，只能靠结果发现。
    # 少了 zero_grad 就是这一类 —— 它不会崩，只会让 loss 在 6 附近来回震荡。
    if final["val"] > 5.5:
        raise SystemExit(
            "\n[结果异常] 验证集 loss = %.4f，比预期高太多（正常应到 4.3 左右）。\n"
            "  最可能的原因：TODO-2（opt.zero_grad()）没写。\n"
            "  梯度是累加的，不清空的话每一步都在叠加历史梯度，等效学习率越跑越大，\n"
            "  loss 会先降后乱 —— 这正是你现在看到的样子。\n"
            "  能跑 ≠ 跑对了。\n" % final["val"]
        )

    print()
    print("=" * 62)
    print("最终结果")
    print("=" * 62)
    print("  初始 loss（未训练）  = 7.1066  ← 就是 ln(1220)")
    print("  训练集 loss          = %.4f" % final["train"])
    print("  验证集 loss          = %.4f" % final["val"])
    print("  验证集困惑度          = %.2f  ← exp(验证集 loss)" % math.exp(final["val"]))
    print()
    print("  困惑度的意思：模型在每一步都相当于在 %.0f 个字里犹豫不决。" % math.exp(final["val"]))
    print("  从 %.0f 降到这里，说明它确实学到了些结构，不是乱猜。" % cfg.vocab_size)


if __name__ == "__main__":
    main()
