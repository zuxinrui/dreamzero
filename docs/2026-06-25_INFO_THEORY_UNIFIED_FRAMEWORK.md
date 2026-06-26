# 一个母体生万法:ELBO、Information Bottleneck、VAE 与 VIB 的统一数学框架,以及它如何解释 SOE 的维度分裂

---

## 0. 一页纸结论 (the single unifying object)

**整个家族——ELBO / VI / VAE / IB / VIB / β-VAE——都是同一个凸对偶恒等式在不同 `(X, Y, β, 用先验还是后验, 用哪个 f)` 选择下的实例。这个恒等式是 Gibbs / Donsker-Varadhan 变分表示:**

$$
\log \mathbb{E}_Q\big[e^{g(X)}\big] \;=\; \sup_{P}\Big\{\, \mathbb{E}_P[g(X)] - \mathrm{KL}(P\,\|\,Q) \,\Big\}, \qquad \text{maximizer } dP^\star \propto e^{g}\,dQ.
$$

读法一句话:**"log-配分函数(cumulant)与 KL 互为 Fenchel 共轭;最优解永远是 Q 的指数族倾斜(exponential tilt)"** [R3, R4, R7]。

- 把 $g=\log p(x\mid z)$、$Q=p(z)$ 代入 → 这就是 **ELBO**,gap 恰好是 $\mathrm{KL}(q\|p(z\mid x))$。
- 把目标写成 $\min \text{(distortion)} + \beta\cdot\text{(rate)}$,rate $=I(X;Z)=\mathbb{E}_x\,\mathrm{KL}(q(z\mid x)\|r(z))$ → 这就是 **rate-distortion / IB / VIB / β-VAE** 的同一条 Lagrangian,β 是 R-D 曲线的对偶斜率 $-\beta=dR/dD$。
- rate 项之所以是"压缩",根子在 **MI 就是一个 KL**:$I(X;Z)=\mathrm{KL}(P_{XZ}\|P_X\otimes P_Z)$ [R1];之所以"代码再处理也不能凭空造信息",根子在 **DPI**:$Y\!-\!X\!-\!Z$ 的 Markov 链强制 $I(Z;Y)\le I(X;Y)$ [R1]。

**对 SOE Fig.5(Q1)的回答**:对角高斯编码器使 rate 在坐标上**可加(tensorize)** [R1],每一维独立面对"花 rate 换 relevance 是否划算"的 R-D 取舍。该取舍的凸最优在每一维上是**门限式**的:relevance 的二阶曲率(Fisher 曲率)若超过 $\beta$,该维就"开"(高 SNR,被使用),否则它最便宜的驻点就是先验本身($\mu_i\!\to\!0,\sigma_i\!\to\!1$,KL$_i\!\to\!0$,SNR$\!\to\!0$,坍塌)。这与 VAE posterior collapse、β-VAE 选择性编码是**同一个机制**。

**对 StableVLA(Q2)的诚实回答**:**结构性类比,不是同一个定理(structural analogy, not theorem-correspondence)。** StableVLA 的 Prop 3.1 是一个构造性代数等式(IB 更新 ≡ channel attention),它**没有 SNR、没有 per-channel rate、没有维度计数或集中不等式**;其门控关闭靠的是协方差启发式 + 消融实验,不是被证明的信息界。两者共享的只是"同出 IB Lagrangian"这一层。下面逐项展开。

---

## 1. 统一的数学母体 (the unifying framework)

这一节把 Duchi 书里散落各章的工具拼成一张图。它们看似七件事,实则一件事的七个切面。

### 1.1 凸对偶:Fenchel 共轭、Fenchel-Young、Bregman [R7, R5a]

一切的原子是 **Legendre-Fenchel 变换**:

$$
f^*(s) := \sup_x \{\langle s,x\rangle - f(x)\}.
$$

$f^*$ 对任何 $f$ 都是闭凸的(仿射函数的上确界)。对闭凸 $f$ 有 **Fenchel-Moreau** $f^{**}=f$(共轭是对合)[R7, Thm C.2.1]——**这是"变分表示能取到等号而非仅是上界"的根本原因**:对偶变量上的 sup 能精确重构原函数。

由定义立得 **Fenchel-Young 不等式**:

$$
\langle s,x\rangle \le f^*(s)+f(x), \qquad \text{等号} \iff s\in\partial f(x) \iff x\in\partial f^*(s).
$$

这一行就是所有变分界的母体。其"gap"就是 **Bregman 散度**

$$
D_f(x,x_0) = f(x)-f(x_0)-\langle\nabla f(x_0), x-x_0\rangle \ge 0.
$$

**为什么对算法设计要紧**:(i) 任何难算的凸量(log-evidence、KL、MI、rate)都能换成"对一个辅助变分变量取 sup"$f(x)=\sup_s\{\langle s,x\rangle-f^*(s)\}$——ELBO、MINE/NWJ/InfoNCE、VIB rate bound、Donsker-Varadhan 全是这一招 [R7]。(ii) 最优解直接由共轭梯度读出 $\partial f^*(s)=\arg\max_x\{\langle s,x\rangle-f(x)\}$;对 entropy 型 $f$ 这自动给出**指数族 / Gibbs / softmax** 形式——这就是为什么最优 IB 编码器、max-ent 模型、energy-based posterior 都长成 $\exp(-\beta\cdot\text{cost})$ [R7, Cor C.2.4]。(iii) **强凸 ↔ 光滑对偶**(Prop C.2.6):$\lambda$-强凸的罚项给出 $(1/\lambda)$-Lipschitz 的对偶映射——这正是"加大 KL/rate 权重 β 既稳定表示又抬高'某维被使用'的门限"的精确表述。

### 1.2 KL 的变分表示:Donsker-Varadhan / Gibbs 变分原理 [R3, R4]

**这是整个家族的种子。** 对共同空间上的 $P,Q$:

$$
\boxed{\;\mathrm{KL}(P\|Q) = \sup_g\big\{\mathbb{E}_P[g] - \log\mathbb{E}_Q[e^g]\big\}\;}\qquad\text{(取到等号于 } g^\star=\log\tfrac{dP}{dQ})
$$

及其对偶半边(自由能形式):

$$
\log\mathbb{E}_Q[e^g] = \sup_P\big\{\mathbb{E}_P[g] - \mathrm{KL}(P\|Q)\big\}, \qquad dP^\star \propto e^g\,dQ.
$$

精确地说:**KL 是 log-MGF / cumulant 泛函的 Fenchel 共轭**(因为 $\log\sum_j e^{x_j}$ 是负熵的共轭)[R3, p.137]。

这一个恒等式是 **ELBO、Gibbs posterior、KL-regularized RL/RLHF、VIB Lagrangian、PAC-Bayes** 的共同祖先。设计一个带信息正则的世界-动作模型,你做的全部选择只是:挑 energy $g$、挑参考 $Q$、挑 KL 缰绳的强度——其余都是记账 [R3]。ELBO 就是把这个不等式读成"$\log p(x)\ge \mathbb{E}_q[g]-\mathrm{KL}(q\|\text{prior})$",其变分 gap 等于 $\mathrm{KL}(q\|p(z\mid x))$。

### 1.3 三位一体:指数族 ↔ 最大熵 ↔ log-loss,且 KL = log-配分函数的 Bregman 散度 [R2b, R5b]

指数族 $p_\theta(x)=h(x)\exp(\langle\theta,\phi(x)\rangle-A(\theta))$,log-配分函数 $A(\theta)=\log\int h\,e^{\langle\theta,\phi\rangle}d\mu$。三条核心事实:

1. **$A$ 凸,$\nabla A(\theta)=\mathbb{E}_\theta[\phi]$(均值参数),$\nabla^2 A(\theta)=\mathrm{Cov}_\theta(\phi)=$ Fisher 信息** [R2b]。优化、统计、信息在同一个 Hessian 上汇合。
2. **指数族内的 KL 恰是 $A$ 的 Bregman 散度**(Prop 3.3.1):$\mathrm{KL}(P_\theta\|P_{\theta+\Delta})=D_A(\theta+\Delta,\theta)$;局部 $\approx \tfrac12\Delta^\top\nabla^2A\,\Delta$,即 Fisher 二次型。
3. **三个程序是同一个凸问题**(Thm 14.4.7 + Cor 14.4.8 + Prop 14.4.12)[R5b]:在矩约束下**最大熵** = 指数族中**最大似然** = **最小 log-loss**。证明核心是那条加减一个模型密度的恒等式 $H(P)=-\mathrm{KL}(P\|P_\theta)+H(P_\theta)$——**与 ELBO 的 $\log p=\text{ELBO}+\mathrm{KL}(q\|p)$ 是同一个代数动作**。

关键边界现象(对 Q1 重要):当均值参数 $\mu$ 落在 marginal polytope 的**边界**(退化 / 近零方差方向),自然参数 $\theta_n\to\infty$ 沿该方向跑,精度矩阵退化成**伪逆** $K=\mathrm{Cov}(X)^\dagger$,把秩亏方向**清零** [R5b, Prop 14.4.1/14.4.3]。这正是"变分模型把无用低方差维坍塌到先验、只保留活跃子空间"的精确机制。

### 1.4 f-散度与数据处理不等式 (DPI):IB 的脊梁 [R1]

$$
D_f(P\|Q):=\int q\,f(p/q)\,d\mu,\quad f\text{ 凸},\ f(1)=0.
$$

KL($f=t\log t$)、reverse-KL、TV($f=\tfrac12|t-1|$)、Hellinger、$\chi^2$ 都是换一个生成元 $f$。靠 perspective transform $f_{\text{per}}(t,u)=u\,f(t/u)$ 它们**联合凸**,这是 Jensen/ELBO gap 与 f-散度变分表示背后的引擎。

**DPI 是 IB 的结构性根基**:任何 Markov 核 $K$ 下 $D_f(K_P\|K_Q)\le D_f(P\|Q)$(Prop 2.2.13),特例 $X\!-\!Y\!-\!Z$ 给出 $I(X;Z)\le I(X;Y)$。这就是 IB 的 Markov 链 $Y\!-\!X\!-\!Z$ 强制 $I(Z;Y)\le I(X;Y)$ 的全部理由:**一个 latent 永远不可能比它的输入更有信息——编码出来的不可能比编进去的多。** 这是 $\min I(X;Z)-\beta I(Z;Y)$ 的天花板,也是后面解释 Table II"内禀维不随 nominal d 增长"的硬约束来源。

散度比较不等式是一套"货币兑换"[R1]:Pinsker $\|P-Q\|_{TV}^2\le\tfrac12\mathrm{KL}$,Bretagnolle-Huber,$2d_{\text{hel}}^2\le\mathrm{KL}\le\log(1+\chi^2)$。**控住一个 KL/MI(VIB 的 rate 项),自动控住一切可操作量(测试误差、估计风险)。**

### 1.5 rate-distortion / redundancy-capacity 对偶:IB Lagrangian 的运作意义 [R6]

把"用模型 $Q$ 给真实源 $P$ 编码"的多余比特叫 **redundancy**,它**恰等于 KL**:$\mathrm{Red}_n(Q,P)=\mathrm{KL}(P\|Q)$ [R6]。于是 IB/VIB 的 rate 项 $I(X;Z)$ 不是随手加的正则,而是"编码 $X$ 经过 $Z$ 花掉的真实比特数",其操作意义由 Kraft-McMillan / 源编码定理($H(X)\le \mathbb{E}[\ell]\le H(X)+1$,最优码长 $\ell=\log 1/p$)钉死 [R2a]。

**核心对偶(Cor 19.3.2,redundancy/capacity duality)**:

$$
\sup_{\pi}\, I(T;X) \;=\; \inf_Q\,\sup_\pi \int \mathrm{KL}(P_\theta\|Q)\,d\pi(\theta).
$$

**信道能携带的最大信息(对输入先验取 sup 的 MI)= 最优通用码的最坏情形 redundancy。** 这给了 IB rate 项一个硬上界——capacity $C$。**一旦 bottleneck 被逼到 capacity 以下,多出来的维度必然坍塌**,这是 SOE Table II(d:16→64 内禀维不变)的理论根。再加 Clarke-Barron 的渐近 redundancy $\tfrac{d}{2}\log\tfrac{n}{2\pi e}+\int\pi\log\frac{\sqrt{\det J_\theta}}{\pi}$ [R6, Thm 19.5.1]:**每个可被数据识别的方向值约 $\tfrac12\log n$ 比特,按 Fisher 信息行列式 $\sqrt{\det J}$ 加权**——Fisher 近零(似然平坦)的方向贡献 ≈ 0,rate-最优码不在它们身上花任何比特,它们坍塌到先验。

### 1.6 互信息与泛化:information usage bounds generalization [R3, S2]

**Donsker-Varadhan → PAC-Bayes → MI 泛化界**,是一条直链 [R3]。核心结果(Cor 6.2.8):学习规则先看样本 $X_1^n$,再产出 $F\sim\pi(\cdot\mid X_1^n)$;取先验为样本边际 $\pi_0=\mathbb{E}[\pi(\cdot\mid X_1^n)]$,则 $\mathbb{E}[\mathrm{KL}(\pi(\cdot\mid X_1^n)\|\pi_0)]=I(F;X_1^n)$,于是

$$
\mathbb{E}\big[(P_nF-PF)^2\big]\ \lesssim\ \frac{\sigma^2}{n}\,I(F;X_1^n).
$$

**压缩即泛化**:限制输出对训练集的信息 $I$,泛化 gap 自动按 $I/n$ 收紧 [S2, Thm 6.3.2]。这给 VIB rate 项一个非启发式的身份:$\mathrm{KL}(\text{posterior}\|\text{prior})$ **同时是**(a)泛化界,(b)比特率,(c)权重/表示正则。取高斯后验/先验时它就是 ridge 罚 $\|\theta\|^2/(2\tau^2)$ [R3, Prop 6.2.7]——和 VAE/VIB 的闭式高斯 KL 是同一代数。

### 1.7 同一母体的其他切面:Fisher / 信息几何、mirror descent、minimax/Fano [S6, S5, S3]

- **Fisher 信息 = 局部几何**:$\mathrm{KL}(P_\theta\|P_{\theta+v})=\tfrac12 v^\top J(\theta)v+o(\|v\|^2)$,且**所有二次可微 f-散度局部都是同一个 Fisher 椭球**,只差标量 $f''(1)/2$ [S6, eq 13.1.8]。所以 VAE/VIB 换 forward-KL / reverse-KL / Hellinger,**全局行为变、但局部 latent 几何与"哪些维被用"不变**——这预言了 SNR 分裂对散度选择的鲁棒性。Van Trees 给出 $\text{后验精度}=\mathbb{E}_\pi[J(\theta)]+J(\pi)$,数据-Fisher 与先验-Fisher **相加**——这正是 VIB 每坐标"似然精度 + 先验/rate 精度"的角力。
- **Mirror descent = Bregman**:KL 就是负熵的 Bregman 散度 [S5];"线性损失 + (1/η)KL-到-锚点"的近端步有 softmax 闭式解——ELBO、IB、VIB、exponentiated gradient 是同一个近端步。Pinsker 等价于"负熵在 $\ell_1$ 下 1-强凸"——信息不等式与优化曲率是一回事。
- **Minimax / Fano**:$I(V;X)=\sum_v\pi(v)\mathrm{KL}(P_v\|\bar P)$,**任意-$Q$ 上界** $I(V;X)\le\int\mathrm{KL}(P_v\|Q)\,d\mu$(任意参考 $Q$,丢掉非负的 $\mathrm{KL}(\bar P\|Q)$)[S3, eq 12.2.1]——这与 VIB rate bound $I(X;Z)\le\mathbb{E}_x\mathrm{KL}(q(z\mid x)\|r(z))$ **结构完全相同**。Assouad 把 $d$ 维问题拆成 $d$ 个独立二元 SNR 测试——这是"某些维被用、其余坍塌"最干净的形式骨架。

---

## 2. ELBO / VI / VAE / IB / VIB / β-VAE 的来源与生动解释 [O1]

每条都是一个短故事:谁、何时、为什么、关键方程、那一下"啊哈"。

**① ELBO / 变分推断(1993,Hinton & van Camp;1998 Neal & Hinton;1999 Jordan 等)。** 故事:你想算 $\log p(x)$,但后验积不出来。于是用一个能算的代理 $q$ 把 $\log p(x)$ 夹起来,而那条 slack——一个 KL——**恰好是你偷懒的价钱,以 nats 计**。
$$\log p(x)=\underbrace{\mathbb{E}_q[\log p(x,z)-\log q(z)]}_{\mathcal{L}(q)}+\mathrm{KL}(q(z)\|p(z\mid x)).$$
啊哈:Hinton 一开始就把 $\mathcal{L}$ 读成**码长**(description length),为后面 bits-back 和 rate-distortion 埋了引线。

**② VAE(2013/14,Kingma & Welling;Rezende 等)。** 故事:把随机性从采样节点**推到外部噪声输入** $z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon$,Monte-Carlo ELBO 就成了能反传的普通网络。
$$\mathcal{L}=\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-\mathrm{KL}(q_\phi(z\mid x)\|p(z)),\quad \mathrm{KL}=\tfrac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1).$$
啊哈:一个 KL-正则的随机瓶颈自编码器;reparameterization 让一切可导。它是 ELBO 的**amortized** 实例,也是 VIB 在 $Y=X,\beta=1$ 的特例。

**③ Information Bottleneck(1999,Tishby-Pereira-Bialek)。** 故事:把 $X$ 挤过瓶颈 $T$,**忘掉除了预测 $Y$ 之外的一切**;relevance 用最少的比特买到。
$$\min_{p(t\mid x)} I(X;T)-\beta I(T;Y),\ \text{s.t. } Y\!-\!X\!-\!T;\quad p(t\mid x)\propto p(t)\,e^{-\beta\,\mathrm{KL}(p(y\mid x)\|p(y\mid t))}.$$
啊哈:表示学习 = 一个 distortion 本身是 MI 的纯 rate-distortion 问题;自洽解是 Gibbs 指数族——与 EM/VI 同一个对象。

**④ Deep VIB(2016/17,Alemi-Fischer-Dillon-Murphy)。** 故事:Tishby 的 Lagrangian 美但不可算;VIB 用 VAE 的同一招把每个 MI 换成可训练的变分界。
$$J_{\text{IB}}=\mathbb{E}_{x,y}\,\mathbb{E}_{z\sim p(z\mid x)}[-\log q(y\mid z)]+\beta\,\mathrm{KL}(p(z\mid x)\|r(z)).$$
啊哈:**综合节点**——把 ③ 的 Lagrangian 倒进 ② 的机器。$\mathrm{KL}(p(z\mid x)\|r(z))$ 就是 VAE 的 rate 项;$Y=X$ 时退化为 β-VAE。SOE 用的正是这个,只是写成 $\max I(Z;A)-\beta I(Z;O)$。

**⑤ β-VAE / "Fixing a Broken ELBO"(2017 Higgins;2018 Alemi 等)。** 故事:单个 ELBO 标量藏着一整条 rate-distortion 前沿;拧 β 就沿前沿滑动,用重构 nats 换压缩 nats。可达区 $H-D\le R$;**很多点 ELBO 相同但表示天差地别**。啊哈:max-likelihood($\beta=1$)**根本钉不住表示**——这正是看维度极化现象的镜片。

**⑥ MDL / Bits-Back(1990 Wallace;1993 Hinton;2004 Honkela-Valpola)。** 故事:变分贝叶斯**本就是一套压缩**。码长 $=-\text{ELBO}=\mathbb{E}_q[-\log p(x,z)]-H[q]$;编码器熵 $H[q]$ 是你藏在噪声里的比特退款。啊哈:$\mathrm{KL}(q\|p)$ 就是 rate(超出先验码的多余比特),$-\mathbb{E}_q[\log p(x\mid z)]$ 就是 distortion——闭环。

```
血缘图 (lineage)

      Gibbs / Donsker-Varadhan 变分恒等式  +  rate-distortion Lagrangian  L = D + βR
                                  │
        ┌──────── ① ELBO / VI (1993-99) ────────┐
        │            log p = L + KL                │
        │                                          │
        ▼                                          ▼
   ② VAE (2014)  ──── reparam ───►       ⑥ MDL / Bits-Back (1990-2004)
   amortized, Gaussian KL              码长 = D + R = −ELBO
        │                                          ▲
        │  借机器(reparam+变分界)                  │ 同一个码长
        ▼                                          │
   ③ IB (1999)  ──Lagrangian──►  ④ Deep VIB (2017)  ──Y=X──►  ⑤ β-VAE / RD plane (2017-18)
   I(X;T)−βI(T;Y)              J_IB(可训练)                 沿 R-D 前沿扫 β
```

---

## 3. 它们如何统一 (the punchline)

把第 2 节六个方法叠在一张 **rate-distortion 平面**上:横轴 rate $R=I(Z;X)=\mathbb{E}_x\mathrm{KL}(q(z\mid x)\|r(z))$(latent 保留了多少关于输入的比特),纵轴 distortion $D=-\mathbb{E}[\log p(\text{target}\mid z)]$(重构或预测的码长 = 一个 Bregman/KL gap [R5a, Ex 14.2.11])。所有方法都是 Lagrangian $L=D+\beta R$ 的不同点 / 不同读法:

| 方法 | $X$(被压缩) | $Y$(target) | $\beta$ | 用先验还是后验 | 对应 §1 定理 |
|---|---|---|---|---|---|
| ELBO / VAE | 数据 $x$ | $x$ 本身(自编码) | $1$ | 后验 $q(z\mid x)$ vs 先验 | DV §1.2;高斯 KL §1.3 |
| β-VAE | $x$ | $x$ | $>1$,扫 | 同上 | R-D 平面 §1.5 |
| IB | $x$ | 标签/未来 $y$ | 扫 | 自洽 Gibbs | DPI 天花板 §1.4 |
| VIB | $x$ | $y$ | 扫 | $p(z\mid x)$ vs $r(z)$ | 任意-$Q$ 变分界 §1.6/§1.2 |
| **SOE** | 观测 $o$ | 动作 $a$ | 扫 | $p_\theta(z\mid o)$ vs $N(0,I)$ | per-coord 高斯 KL §1.3 |

**为什么是同一个东西,逐项扣回 §1 定理**:

1. **rate 项是 KL,因为 MI 是 KL**(§1.4):$I(Z;X)=\mathrm{KL}(P_{ZX}\|P_Z\otimes P_X)$;难算就用**任意-$Q$ 变分上界** $I(Z;X)\le\mathbb{E}_x\mathrm{KL}(q(z\mid x)\|r(z))$(§1.6 / §1.2,丢掉非负的 $\mathrm{KL}(\bar q\|r)$,和 §3-Fano 的 eq 12.2.1 同一招)。这就是 VIB/VAE 的可训练 rate 项,$r$ 是先验。
2. **distortion 项是 Bregman/log-loss**(§1.3):$-\mathbb{E}\log q(y\mid z)$ 是 proper log-loss,其超出最优的部分是负熵的 Bregman 散度 = KL,局部二次、曲率 = 条件 Fisher 信息(§1.3, §1.7)。
3. **整条 Lagrangian 就是 §1.2 的自由能对偶 / §1.5 的 redundancy-capacity 对偶**:$\beta$ 是 R-D 曲线的对偶斜率 $-\beta=dR/dD$(§1.1 Fenchel-Young),capacity 是 rate 项的饱和值(§1.5)。
4. **可达性边界由 DPI 给出**(§1.4):$I(Z;Y)\le I(X;Y)$ 是横轴-纵轴可换信息量的硬上界。

**R-D 平面图**:

```
distortion D = −E[log q(y|z)]
  ^
  │·                       ← 高 β:rate 被压死,所有维坍塌(SOE: model collapse)
  │ ·
  │   ·                    每个 (D,R) 点是一个 β;ELBO=−(D+R) 是一条对角线切片
  │     ·_                 ⑤ "Fixing ELBO":同一 ELBO 值上多点,表示却天差地别
  │       ·--·_
  │            ·----·___   ← R-D 前沿;斜率 −β
  │  低 β:rate 不受限,几乎所有维都开(SOE: 纠缠 latent)·----·____
  └────────────────────────────────────────────────────────►  rate R = I(Z;X)
   R=0(纯先验,全坍塌)          R→capacity C(DPI/redundancy 天花板 §1.4/§1.5)
```

SOE 的 β 消融——**过大→坍塌、过小→纠缠**——就是在这条前沿上往两端滑;capacity 天花板解释了为什么超过它的额外维必然坍塌(Table II)。

---

## 4. 回答 Q1:为什么 SOE Fig.5 出现高 SNR 维与噪音维的分裂

这一节把机制做到滴水不漏。**重要前提:SOE 的 Fig.5 / Table II 是经验观察,SOE 本身没有证明任何定理;下面的"深层数学"全部来自 Duchi 书(§1),SOE 只是它的一个实例。** 我已按对抗性审稿的三条裁决修正了下面四处(SNR 对象、KKT 阶数、分裂形状、对角假设),并标出哪些是书里的、哪些是阈值模型的启发式推断。

### 4.1 五步组合(机制骨架)

**第 1 步:rate 是一个 KL-到-先验。** SOE 目标 $\max I(Z;A)-\beta I(Z;O)$ 的 rate 项 $I(Z;O)=\mathrm{KL}(P_{ZO}\|P_Z\otimes P_O)$,用任意-$Q$ 变分上界(§1.2/§1.6)换成 $\mathbb{E}_o\mathrm{KL}(p_\theta(z\mid o)\|r(z))$,$r=N(0,I)$。这正是 SOE 可训练损失里的 $\beta\,\mathrm{KL}(p_\theta(Z\mid o)\|N(0,I))$。

**第 2 步:KL 在坐标上可加(tensorize)。** 对角高斯编码器 $N(\mu(o),\mathrm{diag}(\sigma(o)^2))$ 对乘积先验 $N(0,I)$,由 KL tensorization(§1.3, Prop 2.1.13)rate 是**独立 per-coordinate KL 的和**。rate 里没有跨维耦合;**每一维被单独定价**——这是 per-dim 现象的结构性来源。

**第 3 步:每坐标 KL 的闭式,与 SNR 的正确对象。** 对 $z_i\mid o\sim N(\mu_i,\sigma_i^2)$ 对先验 $N(0,1)$(§1.3, eq 2.1.5 + 高斯熵 log-det):

$$
\boxed{\ \mathrm{KL}\big(N(\mu_i,\sigma_i^2)\,\|\,N(0,1)\big)=\tfrac12\big(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2\big)\ \ge 0,\ =0\iff(\mu_i,\sigma_i)=(0,1).\ }
$$

**⚠️ 审稿修正(math-rigor)**:不要声称这个**单实例 KL** 是 $\mu_i^2/\sigma_i^2$ 的函数,也不要声称对 $\sigma$ 取最优给出 water-filling——**对固定 $\mu_i$,使该 KL 最小的 $\sigma_i^2$ 恰好是 $1$**(此时 KL$=\mu_i^2/2$),这不是 water-filling 分配。SOE 真正度量的 SNR 是一个**对 $o$ 取期望的"信道"量**:

$$
\mathrm{SNR}_i=\frac{\mathrm{Var}_o(\mu_i(o))}{\mathbb{E}_o[\sigma_i^2]}.
$$

付出的 rate 是花在**均值随输入的变动** $\mathrm{Var}_o(\mu_i(o))$ 上的(不是单实例的 $\mu_i$);在这个信道量上,rate-distortion 形式 $r_i\approx\tfrac12\log(1+\mathrm{SNR}_i)$ 与 ARD/water-filling 才合法适用。**(此 $\tfrac12\log(1+\mathrm{SNR})$ 是标准 R-D 背景,并非 Duchi 书所提供——按 evidence-grounding 裁决如实标注。)** 零-rate 态就是先验本身 $(\mu_i,\sigma_i)=(0,1)$,KL$_i=0$,SNR$_i=0$ —— **第 $i$ 坐标的 posterior collapse**。SOE 的 SNR$_i$ 就是这个 on/off 的读数:SNR$_i\to0\iff$ 坐标坐在先验上 $\iff$ rate$_i=0\iff$ 坍塌。

**第 4 步:relevance 收益有界,所以只值得买有限的信息预算。** DPI 天花板 $I(Z;A)\le I(Z;O)\le I(O;A)$(§1.4)封死了可提取的总 relevance;decoder 项 $-\mathbb{E}_z\log q_\phi(a\mid z)$ 的超额是 Bregman/KL(§1.3),且每加一维的边际 relevance 递减。优化者面对的是 rate 的递减回报。

**第 5 步:KKT 最优在每坐标上是门限式的——但是二阶曲率门限,不是一阶斜率门限。** Lagrangian
$$
L=\mathbb{E}\big[-\mathbb{E}_{z\sim p(z\mid o)}\log q_\phi(a\mid z)\big]+\beta\sum_i\tfrac12(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2).
$$

**⚠️ 审稿修正(math-rigor,导数阶数)**:不能用一阶条件 "$-d(\text{distortion})/d(\text{rate}_i)>\beta$"。在先验处 $(\mu_i=0,\sigma_i=1)$,rate 对 $\mu_i$ 的梯度为零($\tfrac{d}{d\mu_i}\tfrac12\mu_i^2=0$),distortion 对 $\mu_i$ 的梯度也为零(relevance 关于 $\mu_i$ 是偶的),一阶比较退化、无信息。正确的开/关判据是**二阶曲率比较**:在先验附近 $L\approx\tfrac12(\beta-c)\,a^2$,其中 $a$ 是均值振幅、$c$ 是该方向上 relevance 的 Fisher 曲率。于是

$$
\boxed{\ \text{坐标 }i\text{ 开(高 SNR,被用)}\iff c_i>\beta;\qquad c_i\le\beta\implies\text{停在先验 }(\mu_i,\sigma_i)=(0,1),\ \text{SNR}_i=0.\ }
$$

先验对所有 $\beta$ 都是驻点,在 $c_i=\beta$ 处通过**叉式分岔(pitchfork bifurcation)**失去最小性。这是 §1.3(KL 的移动代价 $\approx\tfrac12\Delta^\top\nabla^2A\,\Delta$,Fisher 曲率)、§1.7(Fisher 几何)、§1.5(按 Fisher 加权的 redundancy)的合流:**高 Fisher 曲率(大任务协方差)方向便宜地变得可区分 → 被用;平坦 / 近奇异方向(非极小族,$\mathrm{Cov}$ 非 PD,边界伪逆坍塌 §1.3)无法变得可区分 → 坍塌到先验。**

### 4.2 分裂的形状:是"零尖峰 + 连续尾",不是干净的双峰

**⚠️ 审稿修正(math-rigor,分裂形状)**:凸罚项交付的是一个**精确的零簇**(坍塌维钉在先验,KL$_i=0$)**加上一条连续的正-SNR 尾**(SNR 随 relevance 曲率平滑上升,如 $0.50,1.01,2.04,\dots$)。**真正的直方图双峰(SOE Fig.5 画成两团)还额外需要任务的 relevance/Fisher 特征谱里有一个谱隙(spectral gap)——这是数据的经验性质,不是凸罚项本身的产物。** 因此正确表述是:阈值分岔保证"零尖峰 + 门限以上的连续簇";正部是否双峰是经验问题,且 SOE Fig.5 的双峰是经验观察。

### 4.3 per-coordinate 解耦需要 decoder 局部各向异性(对角曲率)

**⚠️ 审稿修正(math-rigor,对角假设)**:rate 项可加,但 distortion 项 $-\mathbb{E}_z\log q_\phi(a\mid z)$ 通过 decoder **耦合各坐标**。per-coordinate KKT/ARD 解耦要求 **relevance Hessian 在编码器的轴对齐基里(近似)对角**。换言之,某维要能坍塌,decoder 必须沿该方向**局部不敏感**(relevance 曲率平坦)——这正是 §1.3 的非极小族 / 伪逆坍塌图景,应当明说,而不是塞进"边际 relevance 递减"里含糊带过。

### 4.4 β 的角色、Table II 的不变性、与 VAE/β-VAE 同源

- **β 是门限旋钮**(§1.1 强凸↔光滑对偶,Prop C.2.6):增大 β 抬高门限 $c_i>\beta$,把坐标逐个关掉;**过大 → 全部低于门限(model collapse,SOE 失败模式);过小 → 门限太低、几乎全开(纠缠 latent,SOE 失败模式)。** 这与 SOE 消融逐字吻合。**(注:"扫 β 产生尖锐相变 / 双峰直方图"是阈值模型的启发式预测,与 SOE 经验 Fig.5 一致,但不是 SOE 定理,如实标注 — evidence-grounding 裁决。)**
- **Table II 不变性(d:16→64 内禀维不变)= capacity 天花板**(§1.4 DPI + §1.5 redundancy-capacity 对偶):有用 rate 的总量被 $I(O;A)$ 封顶,任务的内禀 relevance 预算固定,再多 nominal 维也不抬高它,**多出来的维只是坍塌**。
- **与 VAE posterior collapse / β-VAE 选择性编码同一机制**:三者都是"对角高斯码 → rate 可加 → per-coordinate R-D 门限 → 角解"。Dai & Wipf(2019)证明这种"极化态"(维要么 active 要么 inactive)对任何良态、对角高斯先验/后验的 VAE/VIB 在最优处都是 generic 的。这不是 bug,是 rate-最优的回答:门限下的维不携带可靠可解码的比特(Le Cam/Assouad SNR 测试,§1.7;Fano 信息预算,§1.5),给它们分 rate 是浪费。

---

## 5. 回答 Q2:与 StableVLA proposition 的关系(诚实评估)

### 5.1 忠实陈述 StableVLA 的 Proposition 3.1

StableVLA(Fu et al., 2026,*Towards Robust Vision-Language-Action Models without Extra Data*)只有一个核心形式结果,Prop 3.1(陈述 PDF p.5,Eqs.1-2;证明 Appendix A,p.13-14,Eqs.8-18)。它是一个**构造性代数等式**,**不是界、不是泛化/集中结果、不是 SNR 或维度选择定理**。

设置:把视觉编码器输出 $X_v\in\mathbb{R}^{N\times D}$ 看作 $D$ 个**通道**观测 $c_j\in\mathbb{R}^N$(不是 $N$ 个空间 token)。把视觉-语言对齐写成 IB 问题 $\min_{\phi(Z\mid X_v)} I(X_v;Z)-\beta I(Z;S)$,$S$ 是干净任务码,把 $D$ 个通道聚成 $D$ 个语义组。

**主张**:在高斯 + latent 结构假设下,IB 软分配更新(Blahut-Arimoto 一步)代数化简为通道注意力门控
$$
Z=V\cdot\sigma(\beta\,Q^\top K),
$$
$Q,K,V$ 是 $X_v$ 的线性投影;归一化算子 $\sigma$ 由假设的 latent 先验决定:**categorical → Softmax**(通道竞争、和为 1),**independent-Bernoulli → Sigmoid**(每个通道-簇对独立门控,带可学偏置 $b$ = "off"-态能量 / 激活阈值)。

**证明力学**:取 IB 更新 $q(c\mid j)\propto[p(c)/Z]\exp(-\beta\,\mathrm{KL}[p(s\mid j)\|p(s\mid c)])$;算高斯-高斯 KL 闭式;在 $\epsilon\to0$ 极限、**共享协方差 $\Sigma$**、**归一化中心 $\mu_c^\top\Sigma^{-1}\mu_c=1$** 下,二次型 $(c_j-\mu_c)^\top\Sigma^{-1}(c_j-\mu_c)$ 塌成双线性 $\mu_c^\top\Sigma^{-1}c_j=k_c^\top q_j$(只含 $j$ 的项被吸进配分函数)。识别 $q_j=\Sigma^{-1}c_j,\ k_c=\mu_c,\ v_j\propto c_j$ 即得上式。实现为 IB-Adapter:per-head Gram 矩阵 $G_h=Q_h^\top K_h$ 建模通道间协方差,sigmoid 门 $A_h=\sigma(G_h\tau_h)$,融合为 $Z=\mathrm{MLP}(X)+\tanh(\lambda)\cdot\text{IB-Adapter}(X)$。鲁棒性主张是**定性的**:噪声通道与语义簇协方差低,$\beta k_c^\top q_j$ 小,sigmoid 门 $\approx0$,被抑制而不强制竞争(Sigmoid>Softmax,由 Table 3 经验支持:换成 Softmax 使 LIBERO-corrupted 掉 16.3 分、CALVIN 2.13→0.46)。**没有 SNR 定义、没有 eigen/奇异值谱、没有低秩子空间证明、没有幸存通道计数、没有集中不等式。**

### 5.2 诚实裁决:结构性类比,不是同一个定理

**判据(litmus test,来自 §1.2/§1.6/§1.1)**:一个**深层**连接要求 StableVLA 的**选择量本身**能写成 $\sup_P\{\mathbb{E}_P[g]-\mathrm{KL}(P\|Q)\}$ 或其 rate-distortion 对偶,且有一个 **per-coordinate rate 变量,其消失 *就是* 坍塌**。

按对抗审稿(deep-vs-superficial 裁决)收紧:**不要因为"该层是从 IB 更新导出的"就给 StableVLA 一个"部分通过"。** 必须区分 **derived-from**(真,但肤浅)与 **the-selection-quantity-is-itself-variational**(假)。rate 项 $I(X_v;Z)$ 在化简里被丢进假设、从未出现在门控里;真正的选择器——门 $\sigma(\beta Q^\top K-b)$——**不是** $\sup_P\{\mathbb{E}_P[g]-\mathrm{KL}(P\|Q)\}$ 对象,**不携带 rate**。所以在**选择陈述**这一层,StableVLA **直接不通过** litmus test,不是"部分通过"。"导出处通过、而部署对象不通过"正是肤浅连接的典型签名。

**真正分隔两者的事实:**

1. **不是同一种对象**:SOE 的选择器是一个散度/rate(nats);StableVLA 的是 $[0,1]$ 里的门值。部署层无 per-channel rate、无 SNR。桥梁"低 SNR $\iff$ 低 IB rate $\iff$ 门 $\to0$"需要 StableVLA 一侧有 rate 变量,Prop 3.1 一个都没提供,也没证明任何把门关闭绑到信息量的界。

2. **(最强反驳,deep-vs-superficial 裁决要求上调权重)假设与结论互相矛盾,不只是"削弱结构链接"。** Prop 3.1 的化简需要**共享协方差 $\Sigma$** 和**归一化中心 $\mu_c^\top\Sigma^{-1}\mu_c=1$**。而 SOE 的 Fig.5 恰恰是一个**per-coordinate 方差异质性**的陈述(SNR$_i$ 随 $i$ 变,才有双峰直方图)。共享-$\Sigma$ + 归一化中心**假设掉了** SOE 分裂赖以存在的二阶矩差异结构。**一个定理的前提杀死另一个定理的结论 → 它们不可能是同一个定理。**

3. **β 不同类**:SOE 的 β 是被**扫**的 R-D Lagrange 乘子,产生坍塌/纠缠的相变窗口(并由 §1.5 capacity 天花板解释 Table II 不变性);StableVLA 的 β 是注意力里一个**固定**的 softmax/sigmoid 温度,从不被扫,无相变。

4. **轴 / 随机性 / 认识论地位都不同**:SOE 是随机编码器、$d=16$ 轴对齐 latent 坐标、显式 KL-rate 罚 + 经验 Fig.5;StableVLA 是确定性、$D$ 个特征通道的 Gram 基、导出的函数形式 + Table-3 消融。

5. **SOE 自己也没证定理**(evidence-grounding 裁决):Fig.5/Table II 是经验的;唯一的深层数学是 **Duchi 书的**(§1)。所以"structural-analogy"这个标签在两侧是**误导性地对称**的——SOE 的深层根基是从 §1.4/§1.6/§1.5 借来的;StableVLA 的门控**连借都借不了**,因为它根本没有 rate 变量。

### 5.3 三栏分清:真共享的数学 / 类比 / 不同定理

| | 真正共享的数学(genuinely shared) | 类比(analogy) | 不同的定理(different theorem) |
|---|---|---|---|
| 内容 | 同出 IB Lagrangian $\min I(X;Z)-\beta I(Z;\cdot)$;同一闭式**高斯-高斯 KL**(§1.3, eq 2.1.5);在共享协方差下都塌成双线性/二次型;同属 redundancy-capacity 哲学"只在买得起 relevance 的方向花 rate"(§1.5) | "都产生 used-vs-suppressed 的坐标分裂并归因于鲁棒/高效";"门控像谱滤波"(StableVLA 自己说是比喻) | SOE 分裂 = **per-coordinate KL-rate 门限 + Fisher 曲率分岔**(§4,机理来自 §1.3/§1.5/§1.7);StableVLA Prop 3.1 = **一步 IB 更新 ≡ channel attention 的代数等式**,选择靠学到的 sigmoid 门 + 协方差启发式 + 消融 |
| 地位 | 可验证、书里有据 | 主题性、表层到中等 | 互不蕴含;前提互斥(见 §5.2 第 2 点) |

**结论(连接强度:structural-analogy)**:**不要声称 Prop 3.1 在数学上预测或解释了 SOE 的 SNR 分裂。** 真正解释 SOE Fig.5 的深层数学住在 Duchi 书的 rate-distortion / per-coordinate-KL / Fisher-曲率机器里(§4),StableVLA 的 proposition 只在"同从一个 IB Lagrangian 导出"这一层与之共享;在实际的**选择陈述**层面,它没有 SNR、没有 per-dimension rate、没有被证明的维度计数或集中保证,且其化简假设与 SOE 现象的异质性前提相冲突。

---

## 6. 对做研究/创新的启发 (concrete levers)

把上面的统一变成机器人策略 / 世界模型可以动手拧的旋钮——尽量具体,不泛泛:

1. **用目标 rate 反选 β,而不是网格搜 β。** 既然 $-\beta=dR/dD$ 是 R-D 曲线斜率,且每坐标在 $c_i>\beta$ 时才开(§4),先定你想要的内禀维数 / 目标比特预算,再据 capacity 天花板(§1.5)反推 β。SOE 的 β 消融(过大坍塌 / 过小纠缠)说明盲扫 β 会撞两端;用 capacity 把 β 钉在中间窗口。

2. **per-modality 解耦 rate(decoupled rates)。** rate 在坐标上可加(§4 第 2 步),所以可以给视觉、语言、本体感受各自一套 β 和各自的对角高斯码,按模态的 DPI 天花板 $I(O_m;A)$ 分配比特,而不是一个全局 β 压所有模态。这直接对应 DreamZero 里 video token / action register 的不同信息角色——给 action register 一个更松的 rate(它要保真),给冗余视觉通道一个更紧的 rate。

3. **用 DPI 设计瓶颈的位置,而不是只设大小。** $I(Z;A)\le I(Z;O)\le I(O;A)$(§1.4)是硬天花板:把瓶颈放在"观测已经丢失某些 task-relevant 信号之后"是浪费。先估各级表示的 $I(O;A)$(可用 Brier-information 这种低方差代理 $I_{\text{sq}}=\sum_j\mathrm{Var}(P(Y=j\mid X))$ [R5a] 代替难估的 Shannon MI),把瓶颈放在信息瓶颈真正发生的层。

4. **rate-distortion-aware exploration(IDS/VDS 思路)。** 探索按 loss/information ratio $R=\text{regret}^2/I(A^\star;\text{obs})$ 打分,最小化它给 $\sqrt n$ regret [R8]。MI 难算就用**条件均值损失的方差**作凸代理 $\mathrm{Var}(\mathbb{E}[\ell\mid A^\star])\le 2\sigma^2 I$ [R8]。更关键:线性 bandit 里 ratio $\le d$(内禀维)而非 $\le|\mathcal{A}|$——**探索代价由驱动 reward 的 latent 的内禀维决定**,这与 §4 的"少数高 SNR 维主导"直接共振:**先用 VIB 压出低维 used-子空间,再在该子空间上做信息导向探索。**

5. **用 Fisher / Gram 谱诊断维度坍塌,而不是事后看 SNR 直方图。** $\nabla^2 A=\mathrm{Cov}(\phi)=$ Fisher(§1.3)统一了损失曲率、特征协方差、per-direction SNR。直接对 latent 的 Fisher(或 StableVLA 式 Gram 矩阵 $G_h=Q_h^\top K_h$)做谱分析:小特征值 = 坍塌/噪音维,大特征值 = used 维。**注意**:真正的双峰需要谱里有 gap(§4.2),所以谱分析还能告诉你"这个任务到底有没有清晰的内禀维"——没 gap 就别指望干净的 Fig.5。

6. **学一个 data-geometry-aware 先验,而不是死守 $N(0,I)$。** 各向同性先验正是低 SNR 维坍塌(而非被重参数化利用)的原因之一;Jeffreys / 参考先验(按 Fisher 体积 $\sqrt{\det J}$)是 capacity-achieving 的最坏-情形先验(§1.5)。用 mixture / flow 先验 $r(z)$ 收紧任意-$Q$ 变分界(gap $=\mathrm{KL}(\bar q\|r)$,§1.6),既降 rate 又能把"本会坍塌"的方向重参数化成有用方向。

7. **Sigmoid-vs-Softmax 是先验结构的选择,不是调参。** StableVLA 的 Prop 3.1 的一个真有用的副产品:**categorical 先验 → Softmax(坐标竞争)、independent-Bernoulli 先验 → Sigmoid(独立门控)**。要"抑制噪声通道而不抽干语义通道的能量",就该用独立 Bernoulli 结构(Sigmoid)。这条对任何想做 modality-alignment / 通道去噪的 VLA 都直接可用,且它确实有 IB 推导背书(尽管不涉及 SNR)。

8. **把 rate 项当泛化预算用,不只当压缩用。** Cor 6.2.8(§1.6)说 $\mathbb{E}[\text{gap}^2]\lesssim\frac{\sigma^2}{n}I(F;X_1^n)$:压 latent 对训练数据的信息**可证地**收紧泛化。对 DreamZero 式 action head,压 action/state register 对输入的信息不是省算力,是 sample-complexity 控制——尤其在 behavior cloning / flow-matching 的低-loss 区,self-bounding 给出 $\sim\mathrm{KL}/n$ 的 fast rate(§1.6)。

---

## 参考与出处

**Duchi, *Statistics and Information Theory* (Nov 2025) 章节锚点:**
- [R1] Ch 2.1-2.2(pp.15-33):KL/MI 非负性、MI=KL、链式法则、DPI、f-散度联合凸、Pinsker/Hellinger/χ²、高斯最大熵 eq 2.1.5。
- [R2a] Ch 2.3-2.4(pp.34-44):Le Cam 二点、Fano、Kraft-McMillan、源编码定理(最优码长 = surprisal)。
- [R2b] Ch 3(pp.49-67):指数族、$A$ 凸、$\nabla A$=均值/$\nabla^2A$=Cov=Fisher、KL=Bregman of $A$(Prop 3.3.1)、曲率控可测性(Key Result 7)。
- [R3] Ch 6.1-6.2(pp.136-146):Donsker-Varadhan(Thm 6.1.1)、Gibbs 自由能对偶、PAC-Bayes(Thm 6.2.1)、**MI 泛化界 Cor 6.2.8**、高斯先验→L2 罚(Prop 6.2.7)。
- [R4] Ch 7.1(pp.160-167):DV 对偶对(Cor 7.1.1)、变分 sub-Gaussian(Thm 7.1.2)、CGF-共轭机器(Thm 7.1.9)、Kantorovich-Rubinstein。
- [R5a] Ch 14.1-14.3(pp.404-419):proper loss、广义熵、l-information、Fenchel-Young、Savage 表示、proper loss ⇔ Bregman、cross-entropy=KL+entropy(Ex 14.2.11)。
- [R5b] Ch 14.3-14.4(pp.419-432):负熵=log-配分共轭、softmax=log-sum-exp 的梯度、**max-ent=MLE=min-log-loss**(Thm 14.4.7/Cor 14.4.8/Prop 14.4.12)、高斯伪逆边界坍塌(Prop 14.4.1/14.4.3)。
- [R6] Ch 19.1-19.5(pp.575-594):log-loss game=robust Bayes、redundancy=KL、**redundancy/capacity 对偶(Cor 19.3.2)**、Shtarkov/NML、Clarke-Barron $\tfrac{d}{2}\log n$(Thm 19.5.1)、Jeffreys=capacity-achieving 先验。
- [R7] App C.2(pp.665-671):Fenchel 共轭、Fenchel-Moreau($f^{**}=f$)、Fenchel-Young、共轭梯度求逆(Prop C.2.3/Cor C.2.4)、强凸↔光滑对偶(Prop C.2.6)、Jensen 即共轭。
- [R8] Ch 18.3(pp.546-556):loss/information ratio、信息论 regret 界(Thm 18.3.2)、Thompson/IDS/VDS、线性 bandit ratio ≤ $d$(Prop 18.3.14)。
- 辅助:[S2] Ch 6.3+Ch 8(MI 控泛化 Thm 6.3.2/Cor 6.3.3、KL-稳定性、DP、Rényi-高斯);[S3] Ch 9+12.2(Le Cam/Fano/Assouad、任意-$Q$ MI 上界 eq 12.2.1、global Fano);[S5] Ch 17(KL=负熵的 Bregman、entropic mirror descent=指数倾斜);[S6] Ch 11+13.1(KL=局部 Fisher 二次型、所有 f-散度局部同一 Fisher 椭球、Cramér-Rao 的"con"、Van Trees 数据-Fisher+先验-Fisher 相加)。

**起源论文(origin papers,[O1]):**
- ELBO/VI:Hinton & van Camp 1993(COLT);Neal & Hinton 1998;Jordan-Ghahramani-Jaakkola-Saul 1999(*Machine Learning* 37)。
- VAE:Kingma & Welling 2014(*Auto-Encoding Variational Bayes*, ICLR, arXiv:1312.6114);Rezende-Mohamed-Wierstra 2014(*Stochastic Backpropagation*, ICML, arXiv:1401.4082)。
- IB:Tishby-Pereira-Bialek 1999(Allerton, arXiv:physics/0004057);Tishby & Zaslavsky 2015(IEEE ITW)。
- VIB:Alemi-Fischer-Dillon-Murphy 2017(*Deep Variational Information Bottleneck*, ICLR, arXiv:1612.00410)。
- β-VAE / RD plane:Higgins 等 2017(*β-VAE*, ICLR);Alemi 等 2018(*Fixing a Broken ELBO*, ICML, arXiv:1711.00464);维度极化的 generic 性:Dai & Wipf 2019(*Diagnosing and Enhancing VAE Models*, ICLR, arXiv:1903.05789)。
- MDL/Bits-Back:Wallace 1990;Hinton & van Camp 1993;Honkela & Valpola 2004(*IEEE TNN* 15(4):800-810)。

**被讨论的对象:**
- SOE:Jin et al. 2025,*Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration*(VIB,$\max I(Z;A)-\beta I(Z;O)$;SNR$_i=\mathrm{Var}(\mu_i)/\mathbb{E}[\sigma_i^2]$;Fig.5、Table II 为经验结果)。
- StableVLA:Fu et al. 2026,*Towards Robust Vision-Language-Action Models without Extra Data*,**Prop 3.1**(p.5 Eqs.1-2;证明 App A,pp.13-14,Eqs.8-18):IB 软分配一步更新 ≡ channel attention $Z=V\cdot\sigma(\beta Q^\top K)$,categorical→Softmax / Bernoulli→Sigmoid;无 SNR、无 per-channel rate、无谱/集中证明。

**诚实性声明**:本报告中所有关于 SOE Fig.5/Table II 的"深层数学"解释,其定理出处是上列 Duchi 书章节,**不是** SOE 或 StableVLA 本身;SOE 的 Fig.5/Table II 是经验观察。StableVLA Prop 3.1 与 SOE SNR 分裂的关系经对抗性核查后判定为 **structural analogy(结构性类比)**,而非定理级对应;且 Prop 3.1 的共享协方差 / 归一化中心假设与 SOE 现象赖以存在的方差异质性前提相冲突。$\tfrac12\log(1+\mathrm{SNR})$ water-filling 形式为标准 rate-distortion 背景,非 Duchi 书所直接提供。"β 扫描产生尖锐相变 / 双峰直方图"为阈值模型的启发式预测,与 SOE 经验 Fig.5 一致但非其定理。
