# -*- coding: utf-8 -*-
"""Generate qwen35_state_probe.ipynb: probe DynamicCache GDN state structure & replaceability."""
import json, uuid

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {"id": uuid.uuid4().hex[:16]},
            "outputs": [], "source": src}

cells = []
cells.append(md([
 "# Qwen3.5 GDN 状态接口确认（DynamicCache 结构 + 可替换性）\n",
 "\n",
 "前置已确认：A100 就绪、transformers 5.22.1 原生支持 Qwen3.5（model_type=qwen3_5）、\n",
 "模型含 `Qwen3_5GatedDeltaNet` 层、`past_key_values` 是 `DynamicCache`（遍历时含 LinearAttentionLayer 对象）。\n",
 "\n",
 "本 notebook 用 Qwen3.5-0.8B（权重已缓存，加载快）确认：\n",
 "1. DynamicCache 内部结构（哪些层存 GDN 循环状态、形状）；\n",
 "2. 能否拿到/替换 GDN 状态张量——这是 LeapQuant/TurboQuant 状态量化注入的前提。\n",
]))

cells.append(code([
 "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
 "import torch\n",
 "name='Qwen/Qwen3.5-0.8B'\n",
 "tok=AutoTokenizer.from_pretrained(name, trust_remote_code=True)\n",
 "model=AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True,\n",
 "                                           torch_dtype=torch.bfloat16).to('cuda')\n",
 "model.eval()\n",
 "print('loaded 0.8B on', torch.cuda.get_device_name(0))\n",
 "print('mem GB', round(torch.cuda.mem_get_info()[0]/2**30,1), 'free /', round(torch.cuda.mem_get_info()[1]/2**30,1))\n",
]))

cells.append(code([
 "# 1) DynamicCache 内部结构\n",
 "import torch\n",
 "inp=tok('The quick brown fox jumps', return_tensors='pt').to('cuda')\n",
 "with torch.no_grad():\n",
 "    out=model(**inp, use_cache=True)\n",
 "p=out.past_key_values\n",
 "print('cache type:', type(p))\n",
 "print('cache len:', len(p))\n",
 "print('pub methods:', [m for m in dir(p) if not m.startswith('_')])\n",
 "for attr in ['_key_cache','_value_cache','key_cache','value_cache','_seen_tokens','seen_tokens']:\n",
 "    if hasattr(p, attr):\n",
 "        c=getattr(p, attr)\n",
 "        print('attr', attr, 'type', type(c).__name__, end='')\n",
 "        try:\n",
 "            print(' len', len(c))\n",
 "            for i, el in enumerate(c):\n",
 "                print('   [', i, ']', type(el).__name__, getattr(el, 'shape', None))\n",
 "        except Exception as e:\n",
 "            print('  iter err', e)\n",
]))

cells.append(code([
 "# 2) 逐索引访问，定位 GDN 状态张量\n",
 "for i in range(len(p)):\n",
 "    try:\n",
 "        el=p[i]\n",
 "        print('idx', i, '->', type(el).__name__, end='')\n",
 "        if hasattr(el, 'shape'):\n",
 "            print('  shape', tuple(el.shape))\n",
 "        elif isinstance(el, (list, tuple)):\n",
 "            print('  len', len(el))\n",
 "            for j, sub in enumerate(el):\n",
 "                print('     sub', j, type(sub).__name__, getattr(sub, 'shape', None))\n",
 "        else:\n",
 "            print('  attrs:', [a for a in dir(el) if not a.startswith('_')][:20])\n",
 "    except Exception as e:\n",
 "        print('idx', i, 'err', type(e).__name__, e)\n",
]))

cells.append(code([
 "# 3) 定位模型中 GDN 层索引，与 cache 层对应\n",
 "gdn=[]; attn=[]\n",
 "for i, m in enumerate(model.model.layers):\n",
 "    if hasattr(m, 'linear_attn'):\n",
 "        gdn.append(i)\n",
 "    elif hasattr(m, 'self_attn'):\n",
 "        attn.append(i)\n",
 "print('GDN layer idx:', gdn)\n",
 "print('attn layer idx:', attn)\n",
 "print('num layers:', len(model.model.layers))\n",
 "# GDN 层在 cache 中的状态如何存取：看 linear_attn 是否暴露 state\n",
 "la = model.model.layers[gdn[0]].linear_attn\n",
 "print('linear_attn type:', type(la).__name__)\n",
 "print('linear_attn state attrs:', [a for a in dir(la) if 'state' in a.lower() or 'recurrent' in a.lower() or 'cache' in a.lower()])\n",
]))

cells.append(code([
 "# 4) 尝试替换 GDN 状态：读当前状态张量，inplace 改动后前向对比\n",
 "import torch\n",
 "# 找到 cache 里 GDN 层的状态位置（DynamicCache 元素）\n",
 "# 先读一次状态值\n",
 "def read_gdn_state(cache):\n",
 "    \"\"\"返回 (indices, shapes) 其中 shape 明显的张量即候选状态。\"\"\"\n",
 "    found=[]\n",
 "    for i in range(len(cache)):\n",
 "        el=cache[i]\n",
 "        if hasattr(el, 'shape'):\n",
 "            found.append((i, tuple(el.shape)))\n",
 "        elif isinstance(el,(list,tuple)):\n",
 "            for j,sub in enumerate(el):\n",
 "                if hasattr(sub,'shape'):\n",
 "                    found.append((i, tuple(sub.shape)))\n",
 "    return found\n",
 "print('candidate state tensors in cache:')\n",
 "for t in read_gdn_state(p): print('  ', t)\n",
 "# 替换验证：把某个候选张量 inplace 置零，重新前向，看是否影响输出（证明它就是被前向读取的状态）\n",
 "try:\n",
 "    inp2=tok('Once upon a time there was a', return_tensors='pt').to('cuda')\n",
 "    with torch.no_grad():\n",
 "        out_ref=model(**inp2, use_cache=True).logits\n",
 "    # 找一个 shape 明显的张量 inplace 置零\n",
 "    cands=read_gdn_state(out_ref.past_key_values)\n",
 "    import torch as T\n",
 "    modified=False\n",
 "    for i,sh in cands:\n",
 "        if len(sh)>=3:  # 状态张量一般是 3-4 维\n",
 "            el=out_ref.past_key_values[i]\n",
 "            if hasattr(el,'shape'):\n",
 "                el.zero_(); modified=True; print('zeroed cache idx', i, sh); break\n",
 "            elif isinstance(el,(list,tuple)):\n",
 "                for j,sub in enumerate(el):\n",
 "                    if hasattr(sub,'shape') and len(sub.shape)>=3:\n",
 "                        sub.zero_(); modified=True; print('zeroed cache idx',i,'sub',j,tuple(sub.shape)); break\n",
 "            if modified: break\n",
 "    if modified:\n",
 "        with torch.no_grad():\n",
 "            out_mod=model(**inp2, use_cache=True).logits\n",
 "        diff=(out_ref-out_mod).abs().max().item()\n",
 "        print('MAX DIFF after zeroing a cache state:', diff)\n",
 "        print('=> if diff>0, that tensor IS consumed by forward (replaceable); if ~0, not used or replaced on re-call')\n",
 "except Exception as e:\n",
 "    print('replace-verify err', type(e).__name__, e)\n",
]))

cells.append(code([
 "# 5) 结论\n",
 "print('== 状态接口确认结束 ==')\n",
 "print('说明：a) GDN 循环状态是否经 past_key_values/DynamicCache 暴露并可按层读写；')\n",
 "print('      b) inplace 修改是否被下一次前向读取（可替换即支持状态量化注入）；')\n",
 "print('      c) 据此设计 LeapQuant(逐窗口) 与 TurboQuant(每步) 的注入点。')\n",
]))

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
path = r"C:\code_2026\turboquant量化\torch_port\qwen35_state_probe.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", path, "cells", len(cells))
