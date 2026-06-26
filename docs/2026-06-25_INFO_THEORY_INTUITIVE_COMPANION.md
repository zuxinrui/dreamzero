# 直觉版伴读:用生活例子读懂《一个母体生万法》

> 这是 `2026-06-25_INFO_THEORY_UNIFIED_FRAMEWORK.md` 的**生动直觉版伴读**。正报告给的是严谨骨架(定理、出处、公式),这一份给的是**手感**——每节配一个生活类比 + 一个带真实数字的小例子 + 一句"啊哈" + 一句扣回正报告的公式。
>
> 章节与正报告一一对应。所有出现的数字都经过独立 agent 复算核对;§4 / §5 的关键 caveat(二阶曲率门限、"零尖峰+连续尾"非干净双峰、StableVLA 是"结构性类比非同一定理")在直觉化时被刻意保留,没有为了顺口而被冲淡。
>
> 读法建议:可单独顺读建立直觉,也可与正报告对照——遇到正报告某节卡住,跳来这里看对应的例子。
>
> 末尾 **§7「深入问答」** 是几轮追问的沉淀,比前文更咬细节:为什么不能直接代入最优倾斜 $P^\star=Qe^g$、VIB 与 ELBO 是巧合还是同源、恒等式左边在经典 IB 里到底是谁、Gibbs 自由能与 Sinkhorn 的 Gibbs 核——并附一张「恒等式 ↔ ELBO ↔ IB/VIB ↔ OT」总表。
>
> **§8「收尾主线」** 把全篇拧成一条可回看的脊:从一个恒等式,经自由能 / DV 对偶 / "$p(x,z)$ 可算但 $\int$ 不可算" / variational inference,一路走到 VAE。想要全局先读 §8。

---

### §0 全局直觉:一个动作,无数化身

**这到底是什么:** ELBO、VAE、IB、VIB、β-VAE 全家桶,其实只有一道菜谱——「拿一个现成的参考分布 $Q$,乘上 $e^{\text{打分}}$,再归一化」——而你做这一下能榨出的汤汁,就是 $\log$-配分函数。换的食材不同(谁是 $X$、谁是 $Y$、$\beta$ 拧多大、参考用先验还是后验、打分函数 $g$ 选哪个),端上桌的菜名就不同,但锅里的动作一模一样。

**一个生活类比:重新加权一群人。** 你手上有一份现成的人口分布 $Q$(比如某城市三类职业的天然占比),你想让它「偏向」某个目标——但你又不想凭空捏造一个全新分布,因为离 $Q$ 太远本身是有代价的(你得花力气、花信息去改造它)。Gibbs / Donsker-Varadhan 恒等式说:最划算的偏移,永远是**保持 $Q$ 的形状不动,逐点乘上 $e^{g(x)}$(给打高分的人按指数加权),再归一化**。

$$\log \mathbb{E}_Q[e^{g}] \;=\; \sup_P\big\{\, \underbrace{\mathbb{E}_P[g]}_{\text{追分}} - \underbrace{\mathrm{KL}(P\|Q)}_{\text{离 }Q\text{ 太远的过路费}}\big\}, \qquad dP^\star \propto e^{g}\,dQ.$$

你在「拼命追高分」和「别离 $Q$ 太远的过路费」之间拔河,最优落点是一个**内部解**——既不躺平在 $Q$,也不全压在最高分那一格。

**一个带数字的例子。** 设三类职业的天然占比 $Q=(0.5,\,0.3,\,0.2)$,某打分函数 $g=(\log 2,\,0,\,\log 4)=(0.693,\,0,\,1.386)$。照菜谱走:
- **乘 $e^{g}$**(即 $\times 2,\times 1,\times 4$):未归一化权重 $=(0.5\times2,\;0.3\times1,\;0.2\times4)=(1.0,\,0.3,\,0.8)$。
- **归一化**:配分函数 $Z=1.0+0.3+0.8=2.1$,于是 $P^\star=(0.476,\,0.143,\,0.381)$。
- **榨出的汤汁** $=\log Z=\log 2.1 = 0.742$ nats。

恒等式说这碗汤 = 追分 − 过路费:$\mathbb{E}_{P^\star}[g]-\mathrm{KL}(P^\star\|Q)=0.858-0.116=0.742$。✓ 分毫不差。

现在看为什么**内部解赢**:躺在 $Q$ 不动,净值只有 $\mathbb{E}_Q[g]-0=0.624$;全压最高分那一格 $(0,0,1)$,虽然追分拉满到 $1.386$,但过路费 $\mathrm{KL}=\log(1/0.2)=1.609$ 把它吃穿,净值 $=-0.223$。两个极端都输给倾斜解的 $0.742$。

**「啊哈」:** 你从来没有「设计」过那个最优分布——你只是选了参考 $Q$、选了打分 $g$,**指数倾斜 $e^{g}\cdot Q$ 自动把它解出来**,而那个 sup 的值就是 $\log$-配分函数。整张「化身表」无非是给同一个动作换名字:

| 化身 | $g$(打分 / energy)是什么 | 倾斜后的 $P^\star\propto Q\,e^{g}$ 长成什么 |
|---|---|---|
| 后验推断 | $g=\log p(x\mid z)$,$Q=p(z)$ 先验 | 贝叶斯后验 $p(z\mid x)$;sup 的值就是 $\log p(x)$(evidence),ELBO 的 gap 是 $\mathrm{KL}(q\|p(z\mid x))$ |
| 二选一 (sigmoid) | 两格上的分差 | logistic 权重 |
| 多选 (softmax) | 各类 logit | softmax 概率 |
| 统计物理 (Boltzmann) | $g=-\beta\cdot\text{能量}$ | Gibbs 分布 $\propto e^{-\beta E}$ |
| KL-正则 RL / RLHF | $g=$ reward / $\beta$,$Q=$ 参考策略 | 倾斜后的最优策略 $\pi^\star\propto\pi_{\text{ref}}\,e^{r/\beta}$ |

VAE/VIB/β-VAE 则是把这道菜插进 rate-distortion 记账:目标写成 $L=\underbrace{D}_{\text{distortion}}+\beta\underbrace{R}_{\text{rate}}$,其中 rate $=\mathbb{E}_x\,\mathrm{KL}(q(z\mid x)\|r(z))$ 就是「离参考先验 $r$ 的过路费」,$\beta$ 是 R-D 曲线的对偶斜率 $-\beta=dR/dD$。拧 $\beta$,就是在「追重构分」和「省压缩比特」之间挪那个内部解的落点。

**扣回报告:** 上面这道菜谱正是报告 §0 的「一页纸结论」与 §1.2 的 Donsker-Varadhan / Gibbs 变分原理($dP^\star\propto e^g\,dQ$,sup 值 $=\log\mathbb{E}_Q[e^g]$)。我这里那枚 $0.742$ 的汤汁,就是 §1.2 里被各家共用的「自由能」。接下来 §1 会把这台机器拆开(Fenchel 共轭、KL tensorize、Fisher 曲率、DPI 天花板),§4–§5 再把它扣到 SOE 的高 SNR 维 / 噪音维分裂、以及 StableVLA 的 channel attention 上——届时你会看到,「过路费 $\mathrm{KL}$ 在坐标上可加,每维各自掂量这趟值不值」正是同一道菜谱的逐维版本。

---

### §1.1 凸对偶:为什么"换个变量取 sup"能精确重构

**这到底是什么**:凸对偶就是同一座山的"两张地图"——一张按位置标高度,一张按坡度标"在这个坡度上山最高能顶多远";来回换两次地图,你拿回的还是原来那座山,一寸不差。正因为不差,所有变分界(ELBO、MINE、VIB rate)才能取**等号**,而不只是 ≤。

**生活类比:山和它的"坡度地图"。** 给你一座山 $f(x)$(横轴是位置 $x$,纵轴是海拔)。Legendre-Fenchel 变换 $f^*(s)=\sup_x\{sx-f(x)\}$ 干的事是:对每一个"坡度" $s$,我拉一条斜率为 $s$ 的直线去托住这座山,记下这条切线在纵轴上被山顶起来的"截距"。把所有坡度的截距画出来,就得到第二张地图 $f^*$——**坡度地图**。神奇之处在于:从坡度地图再做一次同样的变换,你**精确**重建出原山, $f^{**}=f$(Fenchel-Moreau,对闭凸 $f$ 成立)[R7]。这不是近似拼回去,是同一座山的两套坐标,信息零损耗。**"取 sup 能精确重构"的全部秘密就在这条 $f^{**}=f$**——变分界里那个 $\sup$ 不是在凑上界,它在用坡度坐标把原函数一字不差地读回来。

**一个带真实数字的例子:softplus 与它的共轭(二元负熵)。** 取山 $A(\lambda)=\log(1+e^\lambda)$(softplus,log-配分函数的最简形态)。它的坡度 = 导数 = $\nabla A(\lambda)=\mathrm{sigmoid}(\lambda)$。它的共轭 $A^*(p)=p\log p+(1-p)\log(1-p)$,正是**二元负熵**(定义域 $p\in[0,1]$),它的坡度是 $\nabla A^*(p)=\mathrm{logit}(p)=\log\frac{p}{1-p}$。注意 sigmoid 和 logit 互为反函数——**两张地图的坡度刻度互逆**,这就是对偶。

现在拨到坡度 $\lambda=\log 4=1.3863$:
- 山的高度 $A(\log4)=\log(1+4)=\log 5=1.6094$。
- 对应的均值参数 $p=\mathrm{sigmoid}(\log4)=\frac{4}{5}=0.8$(刚好整数比)。
- 共轭地图在那点的值 $A^*(0.8)=0.8\log0.8+0.2\log0.2=-0.5004$。

**Fenchel-Young 等号现场对账**:$\langle s,x\rangle=A(\lambda)+A^*(p)$ 当且仅当 $p=\nabla A(\lambda)$。左边 $\lambda\cdot p=1.3863\times0.8=1.1090$;右边 $A+A^*=1.6094+(-0.5004)=1.1090$。两边**精确**相等,因为我们站在配对点 $(\lambda,p)$ 上。而且 $\mathrm{logit}(0.8)=\log4=1.3863=\lambda$——切点处坡度自洽。

**Bregman 散度 = 当你站错点时的 Fenchel-Young 间隙。** 假如我们偷懒,用同样的坡度 $\lambda=\log4$,却去问一个**错的** $p_2=0.5$:间隙 $A(\lambda)+A^*(p_2)-\lambda p_2 = 1.6094+(-0.6931)-1.3863\times0.5=0.2231>0$。这个正的间隙正是 Bregman 散度 $D_{A^*}(0.5,0.8)=0.2231=\log(1.25)$,而它**恰好就是** $\mathrm{KL}\big(\mathrm{Ber}(0.5)\,\|\,\mathrm{Ber}(0.8)\big)=0.2231$。站对点 → 间隙 0(取到等号);站错点 → 间隙 = 一个 KL 罚款。

**啊哈**:变分界的"≤"和"="不是两回事。$f^{**}=f$ 保证了**存在**一个对偶变量让 sup 顶到等号;Fenchel-Young 告诉你**那一刻是哪一刻**(梯度匹配 $s\in\partial f(x)$);Bregman 散度则把"还没顶到"的差额量化成一个散度——而对 log-配分函数这一族,这个差额**就是 KL**。所以"难算的凸量 → 换成对偶变量取 sup"既能给可优化的界(没顶到时是 ≤),又能在最优处闭合(顶到时是 =),还顺手告诉你次优的代价(KL/Bregman)。顺带:共轭的梯度自动吐出指数族形式——这就是为什么最优 IB 编码器、max-ent 模型、softmax 都长成 $\exp(-\beta\cdot\text{cost})$ [R7, Cor C.2.4]。

**扣回报告**:本节把 [R7] 的三件套坐实了——(1) $f^{**}=f$ 是"变分表示能取等号而非仅上界"的根本原因(报告 §1.1, Thm C.2.1);上面 $A^{**}(\log4)=\lambda p-A^*(p)=1.6094=\log5=A(\log4)$ 的来回正是这条对合;(2) Fenchel-Young $\langle s,x\rangle\le f^*(s)+f(x)$,等号 $\iff s\in\partial f(x)$,其 gap 即 Bregman $D_f\ge0$;我们的 $0.8$ 配 $\log4$ 取等、$0.5$ 配 $\log4$ 漏出 $0.2231$ 的 KL,正是这一行的逐字演示;(3) 由此**每一个变分界都是"把难算的凸量换成对一个对偶变量取 sup"**——ELBO、MINE/NWJ/InfoNCE、VIB rate bound、Donsker-Varadhan 全是 softplus 这个玩具放大后的同一招(§1.1 (i)、§1.2)。下一节 §1.2 的 $\log\mathbb{E}_Q[e^g]=\sup_P\{\mathbb{E}_P[g]-\mathrm{KL}(P\|Q)\}$ 就是把这里的 softplus 换成一般 log-MGF、把二元负熵换成一般 KL 的同一张对偶图。

---

### §1.2 KL 的变分表示:抛硬币把它算到底

**这玩意儿到底是什么**:KL 和 log-配分函数互为 Fenchel 共轭——给一个分布配上一个"评分函数 $g$",最划算的重加权永远是把原分布按 $e^g$ 指数倾斜(exponential tilt),而你能榨出的总收益恰好等于 $\log\mathbb{E}_Q[e^g]$。整个 ELBO/IB/VIB 家族都是这一个恒等式的变装。

#### 比喻:押注一枚公平硬币

你手里一枚公平硬币 $Q=(0.5,0.5)$。有人塞给你一张"赔率表"(就是评分 $g$):押对正面,记 $g=\log 4=1.386$ 个 nat 的奖励;押反面,$g=0$。问题是:你该把多少筹码挪到正面?

挪得越狠,你越能蹭到正面的高奖励——但"偏离公平硬币"本身要交一笔**信息税**,这税就是 $\mathrm{KL}(P\|Q)$。$\sup_P$ 要找的,就是这条"追奖励 − 交税"净收益的最高点。

#### 一路算到底(真实数字)

最优解直接由公式 $dP^\star\propto e^{g}\,dQ$ 读出:

- 正面权重 $\propto 0.5\times e^{\log 4}=0.5\times 4=2$
- 反面权重 $\propto 0.5\times e^{0}=0.5\times 1=0.5$
- 归一化:$P^\star=\big(\tfrac{2}{2.5},\tfrac{0.5}{2.5}\big)=(0.8,\,0.2)$

**第一个"啊哈":这个 $0.8$ 不是拍脑袋,它就是 $\mathrm{sigmoid}(\log 4)=\dfrac{1}{1+e^{-\log 4}}=\dfrac{4}{5}=0.8$。** 指数倾斜在二元情形下精确地就是一个 sigmoid——softmax/sigmoid 门控的"出生证明"。

再看你榨出的"果汁"(log-配分函数):

$$\log\mathbb{E}_Q[e^g]=\log(0.5\cdot 4+0.5\cdot 1)=\log 2.5=0.916.$$

它必须等于"追奖励 − 交税":

$$\underbrace{\mathbb{E}_{P^\star}[g]}_{0.8\times1.386=1.109}-\underbrace{\mathrm{KL}(P^\star\|Q)}_{0.8\ln\frac{0.8}{0.5}+0.2\ln\frac{0.2}{0.5}=0.193}=1.109-0.193=0.916.\ \checkmark$$

#### 懒惰 / 贪婪 / 倾斜:为什么内点最优能赢

| 策略 | $P$ | 追到的奖励 $\mathbb{E}_P[g]$ | 交的信息税 $\mathrm{KL}$ | **净收益** |
|---|---|---|---|---|
| 懒(原地不动) | $(0.5,0.5)$ | $0.693$ | $0$ | $0.693$ |
| 贪(全压正面) | $(1,0)$ | $1.386$ | $\log 2=0.693$ | $0.693$ |
| **倾斜(最优)** | $(0.8,0.2)$ | $1.109$ | $0.193$ | $\mathbf{0.916}$ |

两个极端净收益**一模一样**都是 $0.693$:懒人一分税不交但奖励也少;贪人吃满奖励却被 $\log 2=0.693$ 的税整个吞掉。**唯有内点的 $0.8$ 同时打败两端**——这正是 $\sup_P$ 替你挑中的那个点。

#### 一句话的拉格朗日理由

为什么最优解永远长成 $P\propto Q\,e^{g}$?把约束 $\sum_x P(x)=1$ 用乘子 $\lambda$ 挂上,对 $P(x)$ 求导:$g(x)-(\log\tfrac{P(x)}{Q(x)}+1)-\lambda=0$,直接解出 $P(x)\propto Q(x)e^{g(x)}$。KL 里那项 $\log\tfrac{P}{Q}$ 的导数天生就是个对数,所以最优解天生就是个指数族倾斜——**没有别的形状可选**。

#### 扣回报告

这正是报告 §1.2 那对装框恒等式(自由能半边):

$$\log\mathbb{E}_Q[e^g]=\sup_P\big\{\mathbb{E}_P[g]-\mathrm{KL}(P\|Q)\big\},\qquad dP^\star\propto e^{g}\,dQ.$$

把 $g=\log p(x\mid z)$、$Q=p(z)$ 代进去就是 ELBO,变分 gap 就是 $\mathrm{KL}(q\|p(z\mid x))$;把"追奖励"换成 relevance、"交税"换成 rate,就是 IB/VIB/β-VAE 的同一条 Lagrangian。而它的对偶半边(Donsker-Varadhan 方向)在我们这枚硬币上同样精确:取 $g^\star=\log\frac{dP^\star}{dQ}$(正面 $\log 1.6$、反面 $\log 0.4$),则 $\mathbb{E}_Q[e^{g^\star}]=1$ 故 $\log\mathbb{E}_Q[e^{g^\star}]=0$,于是 $\mathrm{KL}(P^\star\|Q)=\mathbb{E}_{P^\star}[g^\star]=0.193$——KL 被它自己的对数似然比一字不差地重构出来。**这枚硬币,就是整个家族的种子。**

---

### §1.3 三位一体:指数族 ↔ 最大熵 ↔ log-loss

**这玩意儿真正是啥**:一个"带旋钮的基础形状"——拧旋钮(自然参数 $\theta$)就在弯曲分布,旋钮拧到哪儿,由你测到的那几个平均值唯一钉死;而"最大熵 = 最大似然 = 最小 log-loss"是同一个凸问题的三个名字。

**日常类比**:想象一张橡皮膜,平铺时是最平的(均匀分布,最"诚实"、最不预设结构)。现在我只给你一条约束:"它的重心必须在 4.5 处。" 你不能乱捏,只能在"满足这条约束"的前提下让膜尽量保持平整——结果膜会沿一个方向被均匀地、指数地拉斜。这个"拉斜量"就是旋钮 $\theta$。你**没**测的东西(方差、形状细节),最大熵原则替你回答:"那就别假设,保持最平。" 这就是 $p_\theta(x)\propto h(x)e^{\langle\theta,\phi(x)\rangle}$ 的来历——指数族不是某种神秘选择,而是"匹配你测到的矩、其余一律最诚实"的唯一答案。

**一个带真实数字的例子:一颗只知道"平均点数 = 4.5"的骰子**

公平骰子均值是 $3.5$。现在有人告诉你某颗骰子长期平均掷出 $4.5$,除此之外一无所知。最诚实(最大熵)的分布是什么?

约束是"匹配一阶矩 $\mathbb{E}[x]=4.5$",所以充分统计量 $\phi(x)=x$,最大熵解必然是 $p_i\propto e^{\theta i}$——一个几何级数。解出让均值等于 $4.5$ 的旋钮:

$$\theta = 0.371049,\qquad r=e^\theta = 1.449254.$$

于是六个面的概率(每一面是前一面的 $1.449254$ 倍,严丝合缝的等比):

| 面 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $p_i$ | $0.05435$ | $0.07877$ | $0.11416$ | $0.16545$ | $0.23977$ | $0.34749$ |

验算:$\sum_i i\,p_i = 4.5$(分毫不差),且 $p_6/p_1 = r^5 = 6.3933$。熵从均匀的 $\log_2 6 = 2.58496$ bit 掉到 $2.32791$ bit——**恰好下降 $0.25706$ bit**,这 $0.257$ 比特就是"知道均值是 4.5(而非 3.5)"所购买的信息。注意:我只往里塞了一个数字(均值),旋钮就只有一个($\theta$),分布的形状被一阶矩完全决定,没有任何我没测的结构被偷偷塞进去。

(同一台机器换个充分统计量就是 softmax:类别 logits $z_k$ 当自然参数,$p_k = e^{z_k}/\sum_j e^{z_j}$ 就是"匹配 one-hot 期望"的最大熵分类器,log-配分函数 $A=\log\sum_j e^{z_j}$ 就是那个 log-sum-exp。)

**啊哈**:旋钮 $\theta$ 控制的是平均值——这正是 $\nabla A(\theta)=\mathbb{E}_\theta[\phi]$。把旋钮多拧一点点 $\Delta$,分布与原分布的 KL **不是线性而是二次**地长起来:$\mathrm{KL}(P_\theta\|P_{\theta+\Delta})\approx\tfrac12\Delta^\top\nabla^2 A\,\Delta$,而这个 Hessian $\nabla^2 A=\mathrm{Cov}_\theta(\phi)=$ Fisher 信息。三件事在同一个矩阵上汇合:**怎么优化(损失曲率)、统计能分辨多细(Fisher)、信息花多少(KL 的移动代价)是同一个东西**。旋钮在"重心已经很偏"的方向上拧动便宜(曲率大、易分辨 → 值得用),在"平坦"方向上几乎免费却也学不到任何区分度(Fisher≈0 → 坍塌到先验)。

**回扣报告**:报告 §1.3 第 3 条断言"矩约束下最大熵 = 指数族最大似然 = 最小 log-loss 是同一个凸问题",靠的就是恒等式 $H(P)=-\mathrm{KL}(P\|P_\theta)+H(P_\theta)$——和 ELBO 的 $\log p=\text{ELBO}+\mathrm{KL}(q\|p)$ 是同一个"加减一个模型密度"的代数动作。这正是为什么整套框架的标准目标永远是 **"重构 log-loss + KL"**(报告 §3 rate-distortion 平面里 distortion = proper log-loss、其超额部分 = $A$ 的 Bregman 散度 = KL):**最小化 log-loss 就是在做矩匹配,训练 = 把模型的均值参数拧到匹配数据的充分统计量**。而 $\nabla^2 A=\mathrm{Cov}=$ Fisher 这一条,正是 §1.5/§1.7 里"按 Fisher 加权决定哪些维被用、哪些坍塌到先验"的同一个 Hessian。

---

### §1.4 f-散度与数据处理不等式:复印件不会越印越清

**这究竟是什么:** f-散度只是"两个分布差多远"的一族尺子(KL、TV、Hellinger、$\chi^2$ 全是换个生成元 $f$ 量同一件事);而 DPI 说的是——**任何对数据的再加工都只会让这把尺子读数变小,绝不会变大**。

**生动类比——传话游戏 / 复印件。** 你手里有两份原稿,肉眼一看就知道哪份是哪份(差异大)。现在把两份都丢进同一台老式复印机(同一个"信道"),复印件出来都糊了一圈。关键不是"糊"本身,而是:**糊过之后,两份的可区分度只会下降,不会上升。** 你不可能靠"再复印一次"把原稿里本来没印清的字补出来。传话游戏同理:话从第一个人传到第十个人,只会越传越乱,中间没有任何一个人能凭空让信息比上一棒更准。"后处理能模糊,但永远不能锐化"——这就是 DPI 一句话。

**一个带真数字的例子——二元对称信道 (BSC)。** 取两个源分布,都在 $\{0,1\}$ 上:

- $P=(0.9,\,0.1)$(偏向 0)
- $Q=(0.5,\,0.5)$(公平币)

原稿差多远?
$\mathrm{TV}(P,Q)=\tfrac12(|0.9-0.5|+|0.1-0.5|)=0.4$,
$\mathrm{KL}(P\|Q)=0.9\ln\tfrac{0.9}{0.5}+0.1\ln\tfrac{0.1}{0.5}=0.3681\ \text{nats}\ (=0.5310\ \text{bit})$。

现在让两份都过同一台"复印机":翻转概率 $p=0.2$ 的二元对称信道(每个比特有 20% 概率被翻反)。输出:

- $KP=(0.9\cdot0.8+0.1\cdot0.2,\ \dots)=(0.74,\,0.26)$
- $KQ=(0.5,\,0.5)$(公平币过对称信道还是公平币——糊到极致就是它)

复印后还差多远?
$\mathrm{TV}(KP,KQ)=\tfrac12(|0.74-0.5|+|0.26-0.5|)=0.24$,
$\mathrm{KL}(KP\|KQ)=0.74\ln\tfrac{0.74}{0.5}+0.26\ln\tfrac{0.26}{0.5}=0.1201\ \text{nats}\ (=0.1733\ \text{bit})$。

两把尺子都缩了,而且**只缩不涨**:TV $0.4\to0.24$(留下 60%),KL $0.3681\to0.1201$(只留下 32.6%)。TV 的收缩比 $0.24/0.4=0.6$ 还恰好等于 $|1-2p|=|1-0.4|=0.6$——BSC 的标准收缩系数,分毫不差。要是把复印机调到最糊($p=0.5$,纯随机翻转),两个源都被打成 $(0.5,0.5)$,$\mathrm{TV}=0$:信息被洗光,谁也认不出谁。

**啊哈。** 你**没法**通过任何对输出的进一步处理把 TV 从 0.24 拉回 0.4——因为那一步也是个信道,只会再缩。源里被信道抹掉的那部分可区分度,是**永久丢失**的。所有不同的 f-散度(KL、TV、Hellinger、$\chi^2$)在同一台复印机下**齐刷刷一起缩**,这正是它们"联合凸 + perspective transform"的结构在背后保证的(Prop 2.2.13)。顺带:这里 Pinsker 也成立,$\mathrm{TV}^2=0.16\le \tfrac12\mathrm{KL}=0.1840$——控住 KL 就自动控住 TV,这是 §1.4 末尾"货币兑换"的小演示。

**扣回报告的公式。** 把"源分布对"换成"联合分布 $P_{XZ}$ vs 边际积 $P_X\otimes P_Z$",f-散度就变成互信息 $I$;把"复印机"换成 Markov 核,DPI 就给出 $D_f(K_P\|K_Q)\le D_f(P\|Q)$,特例正是报告里反复用的 $I(Z;Y)\le I(X;Y)$。IB 的 Markov 链 $Y\!-\!X\!-\!Z$ 里,latent $Z$ 只是 $X$ 的一份"复印件",所以它**永远不可能比 $X$ 更懂 $Y$**——编码出来的不可能多过编进去的。这就是 $\min I(X;Z)-\beta I(Z;Y)$ 那个无法逾越的天花板,也是 §1.5 capacity 上界和 SOE Table II "$d$ 从 16 涨到 64、内禀维却不变(多出来的维只能坍塌)"背后的硬约束:瓶颈再宽,也榨不出源头没喂进去的信息。

---

### §1.5 rate-distortion / redundancy-capacity:JPEG 质量滑块

**这玩意儿到底是什么**:VIB 的 rate 项 $I(X;Z)$ 不是一个抽象正则,它就是"latent 实际存了多少比特"的发票;而"用错模型 $Q$ 去编码真源 $P$ 多花的比特"恰好就是 $\mathrm{KL}(P\|Q)$——redundancy = KL,一分不多一分不少。

**日常类比:JPEG 的质量滑块。** 你存一张照片,滑块往左拖=文件小(rate 低)但糊(distortion 高),往右拖=清晰(distortion 低)但文件大(rate 高)。对任意给定画质,存在一个"最省字节"的下界,把所有这些点连起来就是 **R-D 前沿**;$\beta$ 就是这根滑块——准确说是前沿的对偶斜率 $-\beta = dR/dD$。你不是在"调参数",你是在沿着一条注定的曲线滑。VIB 训练做的就是这件事:每个 latent 维都在自己的小滑块上独立做"花字节换清晰度划不划算"的决定。

**一个带真实数字的例子:用错码本要罚多少比特。** 设真源有 4 个符号,真实频率 $P=(\tfrac12,\tfrac14,\tfrac18,\tfrac18)$。它的熵 $H(P)=0.5\cdot1+0.25\cdot2+0.125\cdot3+0.125\cdot3=1.75$ bits——这是理论最省码长(给高频符号短码:1、2、3、3 比特)。

现在你偷懒,用一个**错的模型** $Q=(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$(以为四个符号一样常见)去设计码本,于是每个符号都分 $\log_2 4 = 2$ 比特定长码。平均码长 $=E_P[\ell]=2$ bits。

多花的比特 $=2-1.75=0.25$ bits/符号。这恰好就是
$$\mathrm{KL}(P\|Q)=\sum_x p\log_2\frac{p}{q}=0.5\log_2 2+0.25\log_2 1+0.125\log_2\tfrac12+0.125\log_2\tfrac12=0.25\ \text{bits}.$$
**redundancy = KL,数字对得严丝合缝。** 注意 $Q$ 是均匀的,所以 $E_P[\log_2\tfrac1q]=2$ 就是交叉熵,redundancy = 交叉熵 − 熵 = $2-1.75=0.25$。

再看"每个方向值多少比特":Clarke-Barron 说一个**能被数据识别**的参数方向,在 $n$ 个样本下值约 $\tfrac12\log n$ 比特。比如 $n=256$ 个样本,这一维就值 $\tfrac12\log_2 256 = 4$ bits,而且要按 Fisher 行列式 $\sqrt{\det J}$ 加权。反过来,一个**似然平坦**的方向(Fisher≈0,数据怎么变它都不影响 loss),它值 $\approx 0$ 比特——rate-最优码本一个比特都不会浪费在它身上。

**啊哈。** 把 JPEG 滑块拖到最左还有个尽头:就算字节给到无穷,信道本身能传的信息有个天花板 capacity $C$(redundancy/capacity 对偶:$\sup_\pi I(T;X)=\inf_Q\sup_\pi\int\mathrm{KL}(P_\theta\|Q)\,d\pi$)。**任务里真正可学的 relevance 是有限的——多余的 latent 维不是被"压糊",而是根本没有比特可分,它们最便宜的归宿就是坐回先验**($\mu\to0,\sigma\to1$,KL$_i=0$)。这就是为什么把 nominal 维从 16 加到 64,内禀维纹丝不动:你买的不是更大的硬盘,你撞的是信道的物理上限。

**回扣报告。** §1.5 把 VIB 的 rate $I(X;Z)$ 钉成"经过 $Z$ 编码 $X$ 的真实比特数",redundancy $=\mathrm{KL}(P\|Q)$ [R6],其天花板是 capacity $C$(Cor 19.3.2);Clarke-Barron 的 $\tfrac{d}{2}\log\tfrac{n}{2\pi e}+\int\pi\log\frac{\sqrt{\det J_\theta}}{\pi}$ 给出"每个可识别方向 $\approx\tfrac12\log n$ 比特、按 $\sqrt{\det J}$ 加权,平坦方向贡献 ≈0"[R6, Thm 19.5.1]。上面的 0.25-bit 罚单就是 redundancy=KL 的微缩;4-bit/方向是 $\tfrac12\log n$ 的微缩;capacity 天花板正是 SOE Table II(d:16→64 内禀维不变)的理论根——多出来的维必然坍塌。

---

### §1.6 互信息与泛化:背答案 vs 真懂

**这真正在说什么**:你的模型从训练集里"记住"了多少比特($I(F;X_1^n)$),就预付了多大的泛化罚单——gap 的平方按 $I/n$ 计费,一分不多一分不少。

**类比**:两个学生考同一门课。学生 A 把往年真题逐字背下来——他的脑子(输出 $F$)和这套特定卷子(训练集 $X_1^n$)之间互信息巨大;换一套新题就崩。学生 B 只抽出"这章考积分换元"这种通用套路——他对那套具体卷子的互信息很低,但迁移到新题稳如老狗。报告的核心断言(Cor 6.2.8)就是:**A 和 B 谁更会"在新题上翻车",由他们各自吸进去的"针对这套卷子的比特数"精确定价**,而不是由谁更聪明、谁参数更多决定。

**一个带真实数字的例子**。固定 $n=100$ 个训练样本、噪声尺度 $\sigma^2=1$,泛化界是
$$\mathbb{E}[(P_nF-PF)^2]\ \lesssim\ \frac{\sigma^2}{n}\,I(F;X_1^n)=\frac{I}{100}.$$
- **学生 A(背答案)**:$I=1000$ nats → 界 $=1000/100=10$,典型 gap $\sim\sqrt{10}=3.162$。
- **学生 B(懂套路)**:$I=10$ nats → 界 $=10/100=0.1$,典型 gap $\sim\sqrt{0.1}=0.316$。

吸进去的比特少了 $1000/10=100$ 倍,泛化 gap 只缩到 $\sqrt{100}=10$ 倍小($3.162\to0.316$)——**信息是平方级的杠杆,gap 走 $\sqrt{I}$**。

现在把"低信息"这件事翻译成你熟悉的旋钮。取高斯后验 $F=N(\theta,\tau^2)$、高斯先验 $N(0,\tau^2)$,那个 $I$ 就是闭式 KL:
$$\mathrm{KL}\big(N(\theta,\tau^2)\,\|\,N(0,\tau^2)\big)=\frac{\theta^2}{2\tau^2}.$$
要让学生 B 的 $I=10$ nats,在 $\tau^2=1$ 下只需 $\|\theta\|^2=20$(因为 $20/(2\cdot1)=10$)。**这个 $\|\theta\|^2/(2\tau^2)$ 正是 ridge / L2 权重罚**。你以为你在调一个防过拟合的 $\lambda$,其实你在给"模型允许从训练集背走多少 nats"上限。

**啊哈**:VIB 里那一项 rate $\mathrm{KL}(\text{posterior}\,\|\,\text{prior})$ 不是三个不同的东西,而是同一个 KL 戴了三顶帽子——(a)你为压缩 $X$ 花掉的比特数;(b)一张可证的泛化预算单($\le\frac{\sigma^2}{n}\times$ 它);(c)写出来就是 ridge 罚 $\|\theta\|^2/(2\tau^2)$ 的权重正则。压它,三件事一起紧。

**扣回报告**:上面这条直链是 §1.6 的 $\mathbb{E}[\text{gap}^2]\lesssim\frac{\sigma^2}{n}I(F;X_1^n)$(Cor 6.2.8, [R3/S2]),其中取样本边际为先验时 $\mathbb{E}[\mathrm{KL}(\pi(\cdot\mid X_1^n)\|\pi_0)]=I(F;X_1^n)$;高斯先验→L2 罚是 Prop 6.2.7。这也正是 §6 第 8 条对 DreamZero 的用法:压 action/state register 对输入的信息不是省算力,是 sample-complexity 控制——低-loss 区还能拿到 $\sim\mathrm{KL}/n$ 的 fast rate。

---

### §1.7 其他切面:Fisher 几何、mirror descent、Fano

**这仨到底是啥**:同一台机器(KL)的三个旋钮——一把"近处量距离的弯尺"、一台"按正确几何走梯度的步进器"、一个"信息预算决定哪些维学得动"的裁判。

---

#### (a) Fisher = 一把"贴着某点才准"的弯尺

**一句话**:KL 在全局是各种花式曲线,但你把镜头怼到一个点附近放大,所有 $f$-散度都退化成**同一个椭球**,只差一个标量倍率。

**类比**:量地球表面两点距离。隔半个地球,用"大圆距离 / 直线穿地距离 / 弦长",答案差很多(全局形状不同);但量你脚下两块地砖,它们全部塌成同一把**平直卷尺**,只是有人用厘米、有人用英寸——刻度不同,"哪个方向更长"完全一致。

**实算**:两个单位高斯 $P=N(0,1)$、$Q=N(v,1)$,挪动量 $v=0.1$。
- KL$(P\|Q)=\tfrac12 v^2 = \mathbf{0.005}$(Fisher $J=1$,即 $\tfrac12 v^\top J v$)。
- 平方 Hellinger 的局部展开 $H^2 \approx \tfrac18 v^2 = \mathbf{0.00125}$。
- 比值 $0.005/0.00125 = \mathbf{4}$,**恒等于** $4$,与 $v$ 无关——这就是那个标量 $f''(1)/2$(KL 是 $1/2$,平方 Hellinger 是 $1/8$,比值 $4$)。

**aha**:换散度 = 换刻度尺,不换地形。所以你把 VIB/VAE 的目标从 forward-KL 换成 reverse-KL 或 Hellinger,**全局训练行为会变,但"局部 latent 椭球长什么样、哪些维 SNR 高被用、哪些塌成先验"是同一个 Fisher 椭球决定的**——SNR 分裂对散度选择鲁棒。

**对回报告**:§1.7 第一条 $\mathrm{KL}(P_\theta\|P_{\theta+v})=\tfrac12 v^\top J(\theta)v+o(\|v\|^2)$,以及"所有二次可微 $f$-散度局部同一 Fisher 椭球,只差标量 $f''(1)/2$"[S6, eq 13.1.8]。

---

#### (b) Mirror descent = "在正确几何里走梯度",熵步的闭式解又是指数倾斜

**一句话**:当你优化的对象是概率分布,直接欧氏梯度下降会走出单纯形;改用"线性收益 − $\tfrac1\eta$KL-到锚点"的近端步,KL 是负熵的 Bregman 散度,最优解直接是 **softmax**。

**类比**:你要在一个**有限预算的赌盘**上重新分配筹码。直接按收益加减筹码会让某格变负数(非法)。正确做法是"乘性"调整:收益高的格子按 $e^{\text{收益}}$ 放大,再归一化——这就是 §1.1 那个指数倾斜,在优化语境下换了个名字叫熵 mirror descent。

**实算**:锚点是 3 选项的均匀分布 $(\tfrac13,\tfrac13,\tfrac13)$,一步得到的得分(已含步长 $\eta$)$g=(\log 2,\,0,\,0)$。近端步闭式解 $\propto e^{g}$:权重 $(2,1,1)$,归一化 →
$$p^\star=(\mathbf{0.5},\,\mathbf{0.25},\,\mathbf{0.25}).$$
第一格收益高一档($\log 2$),概率精确翻倍;其余两格平分剩下的 $0.5$。

**aha**:ELBO、IB、VIB、exponentiated-gradient **是同一个近端步的不同外衣**。而且 Pinsker 不等式等价于"负熵在 $\ell_1$ 下 1-强凸"——**信息不等式就是优化曲率**,不是两回事。

**对回报告**:§1.7 第二条"KL 是负熵的 Bregman 散度 [S5];线性损失 + $(1/\eta)$KL-到-锚点的近端步有 softmax 闭式解",及 Pinsker ⇔ 负熵 $\ell_1$-强凸。

---

#### (c) Fano / Assouad = 把 $d$ 维难题拆成 $d$ 场独立的"硬币能不能听出偏"

**一句话**:互信息是学习能力的硬预算;Assouad 把一个 $d$ 维估计问题拆成 $d$ 个**互相独立的二元 SNR 测试**,每个方向单独问"这枚硬币的偏倚听得出来吗"。

**类比**:$d$ 个独立的小测验,每个就是"分辨这是偏向正面 $\delta$ 还是偏向反面 $\delta$ 的硬币"。$\delta$ 大的方向(信号强)你稳赢,该维**学得动 → 被用**;$\delta$ 小到接近公平硬币的方向,你只能瞎猜,该维**学不动 → 坍塌到先验**。

**实算**:二元假设 $N(+\delta,1)$ vs $N(-\delta,1)$,看 $\delta$ 怎么决定可学性:
- **强信号** $\delta=0.5$:两分布 KL $=\tfrac12(2\delta)^2=\mathbf{0.5}$;最优(贝叶斯)判错率 $\Phi(-0.5)\approx\mathbf{0.309}$——明显比掷硬币 $0.5$ 好,**这维可学**。
- **弱信号** $\delta=0.1$:判错率 $\Phi(-0.1)\approx\mathbf{0.460}$——离瞎猜 $0.5$ 只差 $0.04$,**这维基本学不动**。

**aha**:Le Cam/Assouad 这套二点法,正是 §4"某些维被用、其余坍塌"最干净的形式骨架——坍塌不是 bug,是裁判判了"这个方向的硬币偏倚低于噪声,押 rate 进去是浪费"。而且互信息的**任意-$Q$ 上界** $I(V;X)\le\int\mathrm{KL}(P_v\|Q)\,d\mu$(丢掉非负的 $\mathrm{KL}(\bar P\|Q)$)和 VIB 的 rate 界 $I(X;Z)\le\mathbb{E}_x\mathrm{KL}(q(z\mid x)\|r(z))$ **结构一模一样**——同一招。

**对回报告**:§1.7 第三条 $I(V;X)=\sum_v\pi(v)\mathrm{KL}(P_v\|\bar P)$、任意-$Q$ 上界 [S3, eq 12.2.1],及"Assouad 把 $d$ 维拆成 $d$ 个独立二元 SNR 测试"。

---

### §2 起源故事:它们从哪来

**这一节真正在讲什么**:六个名字(ELBO / VAE / IB / VIB / β-VAE / Bits-Back)不是六个发明,而是同一个念头在三十年里被六拨人重新撞见——"把一个算不动的量,换成对一个代理分布求 sup,代价是一条 KL"。

**一句话比喻**:想象一个老式复印店,核心机器只有一台——会"按比例倾斜重印"的复印机(就是 §0 那条 Gibbs/DV 恒等式)。六个故事是六个客人,各自带着不同的活儿走进来,但用的是**同一台机器**:有人拿它压缩账本,有人拿它伪造随机性好让反传跑起来,有人拿它"只留能预测明天的那几行字"。机器一直在,只是每次客人给它换了张待复印的纸($g$)、换了张参照原稿($Q$)、换了个倾斜旋钮($\beta$)。

---

**① ELBO / 变分推断(Hinton & van Camp 1993;Neal & Hinton 1998;Jordan 等 1999)。**
活儿:你想知道 $\log p(x)$,但后验积不出来。Hinton 的"啊哈"不是统计的,是**压缩的**——他在 COLT'93 上把 $-\mathcal L$ 直接读成"用模型给数据编码要花多少比特(description length)"。slack 是一条 KL:
$$\log p(x)=\underbrace{\mathbb E_q[\log p(x,z)-\log q(z)]}_{\mathcal L(q)}+\mathrm{KL}\big(q(z)\,\|\,p(z\mid x)\big).$$
KL 永远 $\ge 0$,所以 $\mathcal L$ 是个下界(Evidence Lower BOund)。那条 KL 就是**你偷懒用近似 $q$ 的罚款,以 nats 计**。Jordan 那拨人 1999 把它系统化成"变分推断",成了贝叶斯的主力引擎。

**② VAE(Kingma & Welling 2014;Rezende 等 2014)。**
活儿:ELBO 里要对 $q$ 采样,而采样这个节点**没法反传**(梯度断在随机性上)。那一下灵光:**把随机性从网络里挪出去,变成一个外接的噪声插头**——
$$z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon,\quad \varepsilon\sim N(0,1).$$
现在网络是确定的、$\varepsilon$ 是外部输入,梯度照穿。ELBO 变成一个能 SGD 的普通自编码器:
$$\mathcal L=\mathbb E_{q_\phi}[\log p_\theta(x\mid z)]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,N(0,I)\big),\quad \mathrm{KL}=\tfrac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1).$$
这就是"reparameterization trick"。VAE 是 ELBO 的 amortized 版(用一张编码器网络一次性吐出所有 $x$ 的 $q$),也是后面 VIB 在 $Y=x,\beta=1$ 的特例。

**③ Information Bottleneck(Tishby-Pereira-Bialek 1999)。**
活儿换人:不再是"重建 $x$",而是"把 $X$ 挤过瓶颈 $T$,忘掉除了预测 $Y$ 以外的一切"。压缩与相关性直接对打:
$$\min_{p(t\mid x)}\ I(X;T)-\beta\,I(T;Y),\quad Y\!-\!X\!-\!T;\qquad p(t\mid x)\propto p(t)\,e^{-\beta\,\mathrm{KL}(p(y\mid x)\|p(y\mid t))}.$$
"啊哈":表示学习就是一个 rate-distortion 问题,只是 distortion 本身是一条互信息。注意那个自洽解 $\propto e^{-\beta(\cdots)}$——**又是那台倾斜复印机吐出的指数族**,和 EM/VI 同一个对象。

**④ Deep VIB(Alemi-Fischer-Dillon-Murphy 2017)。**
活儿:Tishby 的 Lagrangian 漂亮,但 $I(T;Y)$、$I(X;T)$ 都算不动。VIB 的动作是**综合**:把 ③ 的 Lagrangian 倒进 ② 的机器,每个 MI 用 VAE 的同一招换成可训练的变分界——
$$J_{\text{IB}}=\mathbb E_{x,y}\,\mathbb E_{z\sim p(z\mid x)}[-\log q(y\mid z)]+\beta\,\mathrm{KL}\big(p(z\mid x)\,\|\,r(z)\big).$$
那个 $\mathrm{KL}(p(z\mid x)\|r(z))$ 就是 VAE 的 rate 项;令 $Y=x$ 它就退化回 β-VAE。SOE 用的正是这台机器,只是写成 $\max I(Z;A)-\beta I(Z;O)$。

**⑤ β-VAE / "Fixing a Broken ELBO"(Higgins 等 2017;Alemi 等 2018)。**
"啊哈"是个坏消息:**单个 ELBO 标量,藏着一整条 rate-distortion 前沿**。$\beta=1$(标准 VAE/max-likelihood)在这条曲线上钉不住任何一点——"很多 $(D,R)$ 点的 ELBO 数值一样,表示却天差地别"。拧 $\beta$ 就是沿前沿滑动,用重构 nats 换压缩 nats。这正是后面看 SNR 维度极化的镜片:你以为在优化一个数,其实在选一条曲线上的位置。

**⑥ MDL / Bits-Back(Wallace 1990;Hinton & van Camp 1993;Honkela-Valpola 2004)。**
最后一位客人其实是第一位 Hinton 的回马枪:**变分贝叶斯本来就是一套无损压缩**。码长 $=-\text{ELBO}=\mathbb E_q[-\log p(x,z)]-H[q]$。最妙的是那个 $+H[q]$:编码器的熵是**退还给你的比特**(你藏在 $z$ 选择的随机性里,接收方解码后能"找回零钱")。于是 $\mathrm{KL}(q\|p)$=rate(超出先验码的多余比特),$-\mathbb E_q[\log p(x\mid z)]$=distortion——闭环回到 ①。

---

**ONE 个带真实数字的例子(用 VAE 的高斯 KL,这是整条血缘里唯一处处可填进真实数字的闭式)。**

取一个二维 latent,编码器对某张图吐出两维。约定先验 $r=N(0,1)$,KL 用 $\tfrac12(\mu^2+\sigma^2-\log\sigma^2-1)$,单位 nats。

- **第 1 维(坍塌维)**:$(\mu_1,\sigma_1)=(0,1)$。
  $\mathrm{KL}_1=\tfrac12(0+1-\log 1-1)=\tfrac12(0)=\boxed{0}$ nats。它**就是先验本身**,一比特没花,什么也没编进去。
- **第 2 维(活跃维)**:$(\mu_2,\sigma_2)=(2,1)$。
  $\mathrm{KL}_2=\tfrac12(4+1-\log 1-1)=\tfrac12(4)=\boxed{2.0}$ nats。
  换算成 bits-back 里真正付的钱:$2.0/\ln 2=\boxed{2.885}$ bits。这就是第 2 维"超出先验码"的多余比特,精确等于 ⑥ 里的 rate。

再看一个反直觉的小验证(④/⑤ 用得上):固定 $\mu_2=2$,问"调 $\sigma$ 能不能少付点 rate?" 对 $\sigma$ 求最小,极小点恰在 **$\sigma=1$**,此时 $\mathrm{KL}=\mu^2/2=2.0$——**收窄方差并不省钱,最便宜的方差就是先验的方差**。所以这一维的 2.0 nats 全是为了"把均值从 0 推到 2"买的可分辨度。

总 rate $=\mathrm{KL}_1+\mathrm{KL}_2=0+2.0=\boxed{2.0}$ nats(② 的 KL tensorize 成逐坐标相加)。整个表示用满 2 维的"名义维度",**有效只用了 1 维**——另一维退款为零、坐死在先验上。

**"aha"**:这两维都喂进同一台复印机,机器一视同仁;但第 1 维没给它任何"该被倾斜"的理由(均值 0、方差 1),倾斜量为 0,rate 为 0,它**自动坍塌成先验**——这不是 bug,是 rate-最优机器的诚实回答:没东西要编,就别花比特。第 2 维有 $\mu=2$ 的位移要保住,机器才肯付 2.0 nats。**"哪些维被用、哪些坍塌"从来不是网络结构决定的,是这条 KL 账单逐维结算出来的。**

**扣回报告的公式/主张**:这个例子是 §2-② 的高斯 KL 闭式 $\mathrm{KL}=\tfrac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$ 的逐项实算;"$\sigma=1$ 才最省、收窄不省钱"正是报告 §4.1 第 3 步的审稿修正(对固定 $\mu$,最小 KL 的 $\sigma^2=1$,**不是 water-filling**);"$\mathrm{KL}_1=0\iff(\mu,\sigma)=(0,1)\iff$ 该维坍塌"就是 §4.1 第 3 步那个零-rate / posterior collapse 判据的最小实例;而把这同一个 $\mathrm{KL}$ 同时读成"码长 / rate / 退款"则是 ⑥ Bits-Back 与 ① ELBO 闭环的字面体现。六个故事、一台机器、一张账单——这正是 §0"一个母体生万法"。

---

### §3 它们如何统一:大家都站在同一条曲线上

**这到底是啥**:ELBO、β-VAE、IB、VIB、SOE 不是六个方法,是**同一条 rate-distortion 曲线上的六个站位**——同一个 $L=D+\beta R$,只是各自把 $\beta$ 拧到了不同刻度。

#### 类比:一队登山者站在同一道山脊上

想象一条南北走向的**山脊线**(就是 R-D 前沿)。山脊每一点都对应一个权衡:往北走(rate $R$ 大)能背更多关于输入的比特、重构/预测更准(distortion $D$ 低);往南走(rate 小)行李轻、但看不清。**$\beta$ 就是每个登山者脚下的坡度计**——$-\beta=dR/dD$,它告诉你"再多花 1 nat 比特,值不值得换来这点 distortion 的下降"。

- ELBO/VAE:坡度计**焊死在 $\beta=1$**,站在山脊正中某个固定点。
- β-VAE:同一个人,但手里能**拧坡度计**,沿脊线来回滑。
- IB / VIB:换了背包内容(target 从"重构自己"换成"预测标签/未来 $y$"),但还是同一道脊。
- SOE:背包装的是"预测动作 $a$",压缩的是"观测 $o$"——**还是这道脊**。

他们看着像在做不同的事,其实只是 $(X\text{ 压谁},\,Y\text{ 预测谁},\,\beta\text{ 拧多大})$ 这三个旋钮拨到了不同位置(对应报告 §3 那张五行表)。脊线本身、脚下的坡度计、"花 rate 买 relevance 划不划算"的逻辑,一模一样。

#### 一个具体例子:拧 $\beta$ 这把旋钮,看维度一个个熄灭

取一个 6 维对角高斯 latent。每一维有一个固定的 **relevance 曲率** $c_i$(Fisher 曲率,衡量"这一维对预测 $y$ 到底有多有用",越大越值钱):

$$c = (3.0,\ 2.0,\ 1.2,\ 0.8,\ 0.3,\ 0.05).$$

§4 的门限判据说得很干脆:**第 $i$ 维"开"$\iff c_i>\beta$**(注意:这是**二阶曲率门限**,不是一阶斜率——在先验处一阶比较退化无信息,真正决定开关的是 relevance 的 Fisher 曲率 $c_i$ 与 $\beta$ 的大小),否则它就坍回先验 $(\mu_i,\sigma_i)=(0,1)$,$\mathrm{KL}_i=0$。现在像拧收音机旋钮一样扫 $\beta$:

| $\beta$ | 开着的维($c_i>\beta$) | 活跃维数 | 山脊上的位置 |
|---|---|---|---|
| $0.5$ | $3.0,2.0,1.2,0.8$ | **4** | 偏南,行李重(纠缠) |
| $1.0$ | $3.0,2.0,1.2$ | **3** | 中段 |
| $1.5$ | $3.0,2.0$ | **2** | 偏中北 |
| $2.5$ | $3.0$ | **1** | 极北,快坍光 |

继续把 $\beta$ 拧过 $3.0$:**一维都不剩,全部坍塌——model collapse**。反过来把 $\beta$ 拧到 $0.05$ 以下:六维全开,**纠缠 latent**。

熄灭的维不是"被随便扔掉",是**精确地钉在先验上**:代入闭式 KL $\;\mathrm{KL}(N(\mu_i,\sigma_i^2)\|N(0,1))=\tfrac12(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2)$,坍塌态 $(\mu,\sigma)=(0,1)$ 给出 $\mathrm{KL}=\tfrac12(1+0-1-0)=0$ nats——一个比特都不花。而一个稍微用上的维 $(\mu,\sigma)=(1,1)$ 花 $\tfrac12\cdot1^2=0.5$ nats。开着的维则各自沿标准 R-D 形 $r_i\approx\tfrac12\log(1+\mathrm{SNR}_i)$ 付费:$\mathrm{SNR}=1$ 付 $0.347$ nats,$\mathrm{SNR}=3$ 付 $0.693$ nats,$\mathrm{SNR}=7$ 付 $1.040$ nats。(此 $\tfrac12\log(1+\mathrm{SNR})$ 是标准 rate-distortion 背景,非 Duchi 书直接提供——见报告 §4 的诚实标注。)

#### 啊哈

**"沿前沿滑动"不是个比喻,是真的在一颗一颗关灯。** 同一组固定的 $c_i$,你只动了 $\beta$ 这一个数,活跃维就从 4 → 3 → 2 → 1 离散地往下掉——因为先验对所有 $\beta$ 都是驻点,在 $c_i=\beta$ 处通过叉式分岔失去最小性。这就是为什么 SOE 的 $\beta$ 消融"过大→坍塌、过小→纠缠"和 VAE 的 posterior collapse 是**同一个旋钮的两端**:他们不是各自踩了不同的坑,是站在同一道脊上、把同一个坡度计拧到了头。(注意:这套门限机制只保证 SNR 直方图上有一个 $\mathrm{SNR}=0$ 的尖峰加一条门限以上的连续正-SNR 尾;SOE Fig.5 那种干净的两团双峰,还额外需要任务的 relevance 谱里有一个经验性的谱隙——而且 SOE 本身只是经验观察,没有证明任何定理,深层数学全在 Duchi 书。)

#### 扣回报告

- 这张"拧 $\beta$ 关灯"表,就是报告 §3 R-D 平面图的离散版:横轴 $R=I(Z;X)=\mathbb{E}_x\mathrm{KL}(q(z\mid x)\|r(z))$,纵轴 $D=-\mathbb{E}[\log q(y\mid z)]$,所有方法都是 $L=D+\beta R$ 上的点,$-\beta=dR/dD$ 是脊线斜率(§3 第 3 条,§1.1 Fenchel-Young)。
- 五行表(ELBO/β-VAE/IB/VIB/SOE)的差别**只在那三列**:压谁($X$)、预测谁($Y$)、$\beta$ 扫不扫——脊线共享(§3 表)。
- 而这道脊往北能走多远不是无限的:**DPI 给了天花板** $I(Z;Y)\le I(X;Y)$,redundancy-capacity 对偶把 rate 项封在 capacity $C$ 以下(§3 第 4 条 + §1.4/§1.5)。这正是上例里"$c_i$ 是固定的一组、再加 nominal 维也变不出新的高曲率方向"的根:多出来的维只能坍塌——也就是 SOE Table II 里 $d:16\to64$ 内禀维不变的硬约束来源。

---

### §4 Q1 直觉:为什么维度会裂成"高 SNR + 噪音"

**这到底是什么**:每个 latent 坐标都是一个独立租客,$\beta$ 是房东每比特的租金;一维只有当它能交付的 relevance "二阶曲率" $c_i$ 大过租金 $\beta$,才租得起这个房间,否则它退租、搬回先验(房租为零、不携带任何信息)。

**生活类比:按坐标交房租。** 想象一栋公寓楼,$d$ 个房间(latent 维)。住进任何房间都要交租:你越想用这个房间存"随输入而变的信息"(把 $\mu_i$ 拉离 0、把 $\sigma_i$ 拉离 1),房东收的租 $\beta\cdot\mathrm{KL}_i$ 越高。每个房间各自算账,互不打扰(对角高斯 → rate 在坐标上可加,§4 第 2 步)。唯一免租的住法是"搬空房间、恢复出厂"——也就是坐回先验 $N(0,1)$。于是每一维独立做一道生意题:**我搬进来能给预测带来的好处,值不值这份租?** 不值,就退租,SNR 归零,变成噪音维。这跟 VAE 的 posterior collapse、β-VAE 的选择性编码是同一台机器。

**一个带数字的算例。** 先把租金表算死。单维 KL 的闭式是
$$\mathrm{KL}\big(N(\mu_i,\sigma_i^2)\,\|\,N(0,1)\big)=\tfrac12(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2),$$
它 $\ge 0$,且**恰好在 $(\mu_i,\sigma_i)=(0,1)$ 处等于 0**——这就是免租的出厂态。注意一个常被搞错的点:固定 $\mu_i=1$ 去挑最省租的方差,最优是 $\sigma_i^2=1$(此时 KL$=0.5$),不是把方差压小。验一下:$\sigma_i^2=0.5\!\to\!0.5966$,$\sigma_i^2=0.8\!\to\!0.5116$,$\sigma_i^2=1.0\!\to\!0.5000$,$\sigma_i^2=1.2\!\to\!0.5088$,$\sigma_i^2=1.5\!\to\!0.5473$——两边都更贵,谷底在 $1$。所以"压低方差去省租"是错的;真正花钱的是**均值随输入的摆动** $\mathrm{Var}_o(\mu_i(o))$,而读出开关的量是 $\mathrm{SNR}_i=\mathrm{Var}_o(\mu_i(o))\,/\,\mathbb{E}_o[\sigma_i^2]$。

现在关键的"开/关判据不是一阶,是二阶"。把 $\beta=1$ 当租金。在出厂态附近,把房间的净亏损展开成均值振幅 $a$ 的二次式:$L\approx\tfrac12(\beta-c)\,a^2$,$c$ 是这一维上 relevance 的 Fisher 曲率。逐个房间算系数 $\tfrac12(\beta-c)$:

- 房间 A,$c=0.5$:系数 $=+0.25>0$ → 出厂态是极小,**退租、坍塌**,SNR$=0$;
- 房间 B,$c=1.0$:系数 $=0.00$ → 临界,正好是叉式分岔点;
- 房间 C,$c=2.0$:系数 $=-0.50<0$ → 出厂态变成山顶,房客**滑出去、租下房间(被使用)**。

为什么必须看二阶?因为在出厂态,租金对 $\mu_i$ 的一阶梯度 $\tfrac{d}{d\mu_i}\tfrac12\mu_i^2$ 在 $\mu_i=0$ 处 $=0$,relevance 对 $\mu_i$ 也是偶函数、一阶 $=0$——一阶账面完全打平,看不出谁该住进来。得拼二阶曲率:$c>\beta$ 才租得起。被租下的那批房间,SNR 平滑爬升,比如沿尾巴排成 $0.50,1.01,2.04,\dots$(对应 rate $r_i\approx\tfrac12\log(1+\mathrm{SNR}_i)$ 分别为 $0.2027,0.3491,0.5559$ nats)。

**aha**:分裂不是谁去人为切一刀分出"重要/不重要",而是一个**叉式分岔(pitchfork)**自动干的——先验对所有 $\beta$ 都是驻点,但当某方向的任务曲率 $c_i$ 一旦越过租金 $\beta$,先验就从谷底翻成山顶,房客被自发"挤出去"。所以你看到的"高 SNR 团 vs 零团",底层是 $c_i$ 与 $\beta$ 的高下,$\beta$ 就是那个调全局门限的租金旋钮:租太贵($\beta$ 太大)→ 全员退租 = model collapse;租太便宜($\beta$ 太小)→ 谁都住 = 纠缠 latent——正是 SOE 消融的两端失败模式。而 Table II 里 $d$ 从 16 加到 64 内禀维却不变,是因为整栋楼能卖出的总 relevance 被 capacity $I(O;A)$ 封顶,加房间只是多空房、多坍塌维。

**两条必须保留的诚实caveat(别简化掉)**:(1) 凸罚项交付的精确形状是"**SNR$=0$ 的尖峰 + 一条连续的正 SNR 尾巴**",不是天然的干净双峰;要看到 Fig.5 那两团,**还需要任务的 Fisher/relevance 谱里恰好有一个谱隙(spectral gap)**,这是数据的经验性质,不是罚项本身产物。(2) 上面"每维独立算账"成立的前提是 decoder 沿坍塌方向**局部不敏感(曲率平坦)**,否则各维通过 distortion 项耦合,per-coordinate 门限不再干净。还要记住:**SOE 自己没证任何定理**,Fig.5/Table II 是经验观察,这套"租金—曲率—分岔"的数学全部借自 Duchi 书(§1),SOE 只是它的一个实例。

**回扣报告**:本节算例就是 §4.1 第 3–5 步的具象化——免租态 $(0,1)$、KL$_i=\tfrac12(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2)$、开关读数 $\mathrm{SNR}_i=\mathrm{Var}_o(\mu_i)/\mathbb{E}_o[\sigma_i^2]$,以及那个加框的二阶判据 "$c_i>\beta\Rightarrow$ 开,$c_i\le\beta\Rightarrow$ 停在先验";三个房间的 $\tfrac12(\beta-c)$ 符号正是 $L\approx\tfrac12(\beta-c)a^2$ 的叉式分岔(§4.1 末);"尖峰+尾、需谱隙"对应 §4.2,"decoder 各向异性前提"对应 §4.3,"$\beta$ 旋钮 / Table II = capacity 天花板"对应 §4.4。

---

### §5 Q2 直觉:和 StableVLA 是表亲,不是同一个定理

**这到底是什么**:SOE 和 StableVLA 长得像、姓氏一样(都姓 "IB Lagrangian"),做的动作也一样("留一些方向、压住另一些"),但一个用的"开关"是以 nats 计的 *rate*,另一个用的是 $[0,1]$ 里的 *门值* —— 是表亲,不是同一个人。

**类比:两个表亲**。想象 IB 家族的家庭聚会。表亲 A(SOE)是个会计:他衡量"这条通道值不值得用",靠的是**记账的比特数**——花了多少 nats 把 latent 推离先验。账上 0 nats = 这条通道根本没开 = 坍塌。表亲 B(StableVLA)是个调音师:他衡量"这条通道开多大",靠的是一个**旋钮位置**——sigmoid 转出来的一个 $[0,1]$ 数。两人都在做"开/关通道",家族遗传的鼻子一样(都从 $\min I(X;Z)-\beta I(Z;\cdot)$ 这条 Lagrangian 下来),所以**外人一看以为是同一个人**。但会计的"0"是一个**测得出来的 rate**,调音师的"0"只是**旋钮拧到了底**,背后没有任何账本。

**一个具体的算例(把两套度量摆在同一张桌上)**。取一条具体通道,SOE 侧编码器给出 $z_i\mid o\sim N(\mu_i,\sigma_i^2)$。
- 若这一维**被用**:比如 $\mu_i=2,\sigma_i=1$。代进报告 §4 的闭式 $\mathrm{KL}=\tfrac12(\sigma_i^2+\mu_i^2-1-\log\sigma_i^2)$,得 $\tfrac12(1+4-1-0)=2$ nats —— **账上写着 2 nats,这维真金白银开着**。
- 若这一维**坍塌**:$\mu_i=0,\sigma_i=1$,得 $\tfrac12(1+0-1-0)=0$ nats —— **账上 0,坍塌的精确定义**。这就是会计的"开关":2 vs 0,单位是 nats,可以直接读出 SNR。

现在看调音师 StableVLA。它的门是 $\sigma(\beta\,k_c^\top q_j)$。假设某通道-簇对的双线性打分 $\beta\,k_c^\top q_j = 0$(噪声通道),门 $=\sigma(0)=0.5$;打分 $=2$,门 $=\sigma(2)=0.881$;打分 $=-2$,门 $=\sigma(-2)=0.119$。**注意这三个数都是 $[0,1]$ 里的旋钮位置,没有一个是 nats,没有一个能告诉你这条通道花了多少 rate。** 你没法问"$0.119$ 这个门对应多少比特"——这个问题在 StableVLA 里没有定义。

**致命的一刀(为什么不可能是同一个定理)**。StableVLA 的 Prop 3.1 证明,需要一个前提:所有通道**共享同一个协方差 $\Sigma$**,且中心被**归一化成 $\mu_c^\top\Sigma^{-1}\mu_c=1$**。把这个前提翻译回 SOE 的语言:它等于**强行规定每一维的方差都一样**。可是 SOE Fig.5 那个零尖峰 + 连续尾的直方图,它存在的全部理由,就是 SNR$_i=\mathrm{Var}_o(\mu_i)/\mathbb{E}_o[\sigma_i^2]$ **逐维不同**——有的维 SNR 高(被用),有的趋近 0(坍塌)。表亲 B 为了证他的等式,第一步就把"每维方差不一样"这件事**假设掉**了;而这恰恰是表亲 A 整个现象赖以存在的土壤。**一个表亲的关键前提,直接抹平了另一个表亲的整个现象**——前提杀结论,所以不可能是同一个定理,顶多是同一个家族。

**"啊哈"**:家族相似 ≠ 同一个定理。两人都"用一些、压一些",都从 IB Lagrangian 下来——这是**结构性类比**(真,但表层)。可一旦你追问"你那个开关是什么单位",会计答"nats / rate / SNR",调音师答"$[0,1]$ 的门值,没有 rate"——而且调音师为了能写出那个门,先把会计赖以工作的方差异质性给抹了。所以正确的判词是:**structural analogy, not theorem-correspondence**,而不是"StableVLA 证明了 SOE 的 SNR 分裂"。

**一个真能用的副产品**(别把孩子和洗澡水一起倒掉):调音师的旋钮**形状**是个真选择。先验是 **categorical → Softmax**(通道之间零和竞争,门加起来 $=1$);先验是 **independent-Bernoulli → Sigmoid**(每个通道独立开关,带可学偏置 $b$ 当激活阈值)。这不是玄学:StableVLA Table 3 给了硬数据——把 Sigmoid 换成 Softmax,**LIBERO-corrupted 掉 16.3 分,CALVIN 从 2.13 跌到 0.46**。要"抑制噪声通道而不抽干语义通道的能量",就该用 Bernoulli 结构(Sigmoid),因为 Softmax 的零和竞争会逼着通道互相抢能量。这条对任何做 modality-alignment / 通道去噪的 VLA 直接可用。

**扣回报告**:对应 §5.2 第 1 点(SOE 选择器是散度/rate(nats),StableVLA 是 $[0,1]$ 门值,部署层无 per-channel rate、无 SNR)与第 2 点("最强反驳":共享协方差 $\Sigma$ + 归一化中心 $\mu_c^\top\Sigma^{-1}\mu_c=1$ 假设掉了 SOE 分裂赖以存在的二阶矩异质性,前提杀结论)。门值算例对应 §5.1 的 $Z=V\cdot\sigma(\beta Q^\top K)$;Softmax/Sigmoid 与 16.3 分、2.13→0.46 对应 §5.1 末尾的 Table 3 经验支持,以及 §6 第 7 条"Sigmoid-vs-Softmax 是先验结构的选择"。最终判词与 §5.3 表格一致:**genuinely-shared math + structural analogy,但 different theorem,互不蕴含、前提互斥**。

---

### §6 研究启发:可以动手拧的旋钮

**这一节 REALLY 是什么:** 前五节把 ELBO/VIB/IB/SOE 证明成同一个 rate-distortion 引擎之后,§6 是引擎舱盖打开后的"旋钮面板"——每个旋钮都对应面板上一条已被证明的物理定律,而不是炼丹时凭手感乱拧的超参。

**生活类比:** 想象你在给 DreamZero 这台机器人做一次搬家。家里 16 个箱子(latent 维)要塞进一辆有限载重的卡车(capacity 天花板 $I(O;A)$)。新手的做法是"先全装上,装不下再说"(网格搜 β)。老手的做法是:先量卡车载重,再按"每件东西对完成任务的边际价值"贴价签,价签低于运费门槛 $\beta$ 的直接留在原地(坍塌到先验)。下面八个旋钮,就是搬家老手的八条操作守则。

---

**旋钮 1 — 用目标 rate 反选 β,别网格搜。**
*So what:* β 不是玄学,它就是 R-D 曲线的斜率 $-\beta=dR/dD$。
*Do this:* 先定你想要的内禀维数。假设你判断 DROID 桌面 pick-and-place 的有效自由度约 8 维,每个被"识别"的方向按 SOE 的 R-D 形式值约 $r_i\approx\tfrac12\log_2(1+\mathrm{SNR}_i)$ 比特。若希望 action register 里高 SNR 维每维各开到 $\mathrm{SNR}=3$(即每维恰 $\tfrac12\log_2 4=1.0$ 比特),目标 rate 就是 $8\times1.0=8$ 比特。把 β 钉在"恰好让第 9 维的 relevance 曲率 $c_9\le\beta$ 关掉、第 8 维 $c_8>\beta$ 开着"的窗口,而不是从 0.001 扫到 100。SOE 的 β 消融(过大坍塌、过小纠缠)正是告诉你盲扫必撞两端。

**旋钮 2 — per-modality 解耦 rate。**
*So what:* rate 在坐标上可加(§4 第 2 步),所以可以按模态切片定价。
*Do this:* 别用一个全局 β 同时压 video token 和 action register。action register 要保真(它直接出 24 步动作),给它松 β,放它开到比如每维 $\mathrm{SNR}=7$($\tfrac12\log_2 8=1.5$ 比特/维);冗余视觉通道(同一桌面三路相机大量重叠)给它紧 β,逼到 $\mathrm{SNR}=1$($\tfrac12\log_2 2=0.5$ 比特/维)甚至坍塌。比特预算按各模态的 DPI 天花板 $I(O_m;A)$ 分,而不是平均主义。

**旋钮 3 — 用 DPI 选瓶颈的"位置",不只是"大小"。**
*So what:* $I(Z;A)\le I(Z;O)\le I(O;A)$ 是硬天花板;在观测已经丢了 task 信号"之后"再卡瓶颈纯属浪费。
*Do this:* 别只问"latent 开 16 维还是 64 维",先估各级表示的 $I(O;A)$。Shannon MI 难估就用低方差代理 Brier-information $I_{\mathrm{sq}}=\sum_j\mathrm{Var}(P(Y{=}j\mid X))$,把瓶颈放在信息真正流失的那一层(比如 VAE 编码之后、DiT 注入 action register 之前)。

**旋钮 4 — rate-distortion-aware 探索。**
*So what:* 探索代价由"驱动 reward 的 latent 内禀维"决定,不由动作空间大小决定——线性 bandit 里 information ratio $\le d$(内禀维)而非 $\le|\mathcal{A}|$。
*Do this:* 先用 VIB 压出低维 used-子空间(就是旋钮 1 里那 8 维),再"只在这 8 维上"做信息导向探索,而不是在原始 8 维关节×连续幅度的全空间瞎撞。这与 SOE 的"少数高 SNR 维主导"和 on-manifold exploration 直接共振:**先压到流形,再在流形上探索。** MI 难算就用条件均值损失的方差作凸代理($\mathrm{Var}(\mathbb{E}[\ell\mid A^\star])\le 2\sigma^2 I$)。

**旋钮 5 — 用 Fisher/Gram 谱诊断坍塌,别事后看 SNR 直方图。**
*So what:* $\nabla^2 A=\mathrm{Cov}(\phi)=$ Fisher 同时是损失曲率、特征协方差、per-direction SNR。
*Do this:* 直接对 action register 的 Fisher(或 StableVLA 式 Gram 矩阵 $G_h=Q_h^\top K_h$)做谱分解。小特征值 = 坍塌/噪音维,大特征值 = used 维。**关键:盯着 gap。** 真正的双峰(Fig.5 那种两团)需要谱里有清晰的 spectral gap——这是数据性质,不是凸罚项变出来的。谱里没 gap,就别指望干净的维度分裂图,也别误诊"训练失败"。

**旋钮 6 — 学一个 data-geometry-aware 先验,别死守 $N(0,I)$。**
*So what:* 各向同性先验正是低 SNR 方向坍塌(而非被复用)的元凶之一;Jeffreys/参考先验(按 Fisher 体积 $\sqrt{\det J}$ 加权)是 capacity-achieving 的最坏情形先验。
*Do this:* 用 mixture 或 flow 先验 $r(z)$ 替代 $N(0,I)$,把任意-$Q$ 变分界收紧(gap $=\mathrm{KL}(\bar q\|r)$)。这既降 rate,又能把"本会坍塌"的方向重参数化成有用方向——本质是把卡车里的死角改装成可用储物格。

**旋钮 7 — Sigmoid vs Softmax 是先验结构,不是调参。**
*So what:* categorical 先验 → Softmax(坐标互相竞争抢能量);independent-Bernoulli 先验 → Sigmoid(逐通道独立门控)。这条有 StableVLA Prop 3.1 的 IB 推导背书(尽管不涉及 SNR)。
*Do this:* 要"抑制噪声通道、又不抽干语义通道能量"——在 modality-alignment / 通道去噪的门控头上,选 Sigmoid(独立 Bernoulli),别无脑用 Softmax,否则强通道会把弱但有用的通道挤死。

**旋钮 8 — 把 rate 项当 sample-complexity 预算,不只当压缩。**
*So what:* $\mathbb{E}[\text{gap}^2]\lesssim\frac{\sigma^2}{n}I(F;X_1^n)$——压 latent 对训练数据的信息,**可证地**收紧泛化。
*Do this:* 在 DreamZero 的 behavior-cloning / flow-matching 训练里,压 action/state register 对输入的 MI 不是省显存,是直接买 sample-complexity:同样 $n$ 条 teleop 数据,$I$ 减半,泛化 gap 的方差上界减半。低-loss 区还有 self-bounding 的 $\sim\mathrm{KL}/n$ fast rate——这对"30min teleop 数据迁移新 embodiment"这种数据稀缺场景尤其值钱。

---

**aha:** 八个旋钮看着杂,其实是同一台引擎的八个出气口。1/2/4/8 都在调"花多少比特"(rate 轴),3 在调"在哪里花"(DPI 位置),5 在"读仪表盘"(Fisher 谱),6/7 在"换油"(先验结构)。一旦你把 β 看成 R-D 斜率、把 KL 看成比特、把 Fisher 看成 SNR,这些就从八个独立超参塌缩成"在 rate-distortion 平面上选一个点 + 选一组坐标基"两件事。

**tie-back:** 这八条逐字对应报告 §6 的 levers 1–8,而每条的"为什么成立"都指回前文:旋钮 1 靠 $-\beta=dR/dD$(§1.1/§3)与门限 $c_i>\beta$(§4.5);旋钮 2 靠 KL tensorization(§4 第 2 步);旋钮 3/4 靠 DPI 天花板 $I(Z;A)\le I(Z;O)\le I(O;A)$(§1.4)与 ratio $\le d$(§1.7/[R8]);旋钮 5 靠 $\nabla^2A=\mathrm{Cov}=$Fisher(§1.3)且双峰需谱隙(§4.2);旋钮 6 靠 Jeffreys=capacity-achieving 先验(§1.5);旋钮 7 靠 StableVLA Prop 3.1(§5);旋钮 8 靠 Cor 6.2.8 的 $\frac{\sigma^2}{n}I$ 泛化界(§1.6)。

---

## §7 深入问答:把恒等式、ELBO、VIB、经典 IB 钉成一张图

> 这一节是几轮追问的沉淀,比前面更"咬细节"。三个问题层层递进:① 既然知道最优解是 $P^\star\propto Qe^g$,为什么不直接代入?② VIB 长得像 ELBO,是巧合还是同源?③ 恒等式左边那个 $\log\mathbb{E}_Q e^g$,在经典 IB 里对应谁?

### §7.1 为什么不能直接代入最优倾斜 $P^\star=Qe^g$?——$Z$ 才是命门

**一句话:** $P^\star\propto Qe^g$ 只给了分布的"形状",没给那个归一化常数;而那个常数恰恰是你算不出来的东西。

写全(以 VAE 为例,$Q=p(z)$、打分 $g=\log p(x|z)$):

$$P^\star(z|x)=\frac{Q(z)e^{g(z)}}{Z}=\frac{p(z)\,p(x|z)}{Z},\qquad Z=\int p(z)\,p(x|z)\,dz=p(x).$$

- 分子 $p(z)p(x|z)=p(x,z)$ 是**联合分布,你本来就完全知道**(先验 × decoder,任意 $(x,z)$ 当场可求值)。
- 所以"$P^\star\propto Qe^g$"翻译过来就是"**后验 ∝ 联合**"——这就是贝叶斯公式本身,没给任何新东西。
- **全部难处缩在 $Z=p(x)$ 这个对 $z$ 的高维积分里。** 写下 $P^\star\propto Qe^g$ 不是解题,是把题**重抄了一遍**。

类比:像一张菜谱,最后一步写着"……然后除以全宇宙的总质量"。前面每步都能做,唯独最后这步做不到。

**用硬币一秒点破——能不能直接代入,只看 $Z$ 能不能算:**

| | 硬币(§1.2) | VAE |
|---|---|---|
| $z$ 空间 | 2 个结果 {正,反} | 连续高维 $\mathbb{R}^d$ |
| $Z=\mathbb{E}_Q[e^g]$ | 两项求和 $0.5\cdot4+0.5\cdot1=2.5$,秒算 | 积分 $\int p(z)p(x\mid z)dz$,积不出 |
| 直接代入 $Qe^g$? | **能**,当场得 $(0.8,0.2)$ | **不能**,卡在 $Z$ |

§1.2 那次我们**真的就直接代入了**,因为 $Z$ 是两项和。VAE 是同一个恒等式,只是 $Z$ 变成积不出的高维积分,所以才退而求其下界。

**恒等式 vs ELBO 在解"不同顺序"的问题(你的直觉对了):**
- **恒等式 = 答案键(characterization)**:告诉你最优解长啥样($Qe^g$)、最优值多少($\log Z=\log p(x)$)——但答案里含着算不出的 $Z$。
- **ELBO = 能真跑的方法(algorithm)**:它的式子 $\mathbb{E}_q[\log p(x|z)]-\mathrm{KL}(q\|p(z))$ 里**从头到尾没有 $Z$**,只有联合 $p(x,z)$(能求值)和你自选的 $q$。你用"优化一个无 $Z$ 的下界"换掉了"精确算 $Z$"。

**还有两层让它更实在:**
- **抬天花板**:gap $\log p(x)-\text{ELBO}=\mathrm{KL}(q\|p(z|x))$。对 $q$ 取 max → 把 encoder 推向后验(**推断**);对 decoder 参数 $\theta$ 取 max → 把整个天花板 $\log p_\theta(x)$ 顶高(**学模型**,让 $p_\theta(x)$ 贴近数据)。一个 ELBO,同时干这两件事。
- **摊销 (amortization)**:VAE 不对每个 $x$ 重解一次优化,而是训**一个 encoder 网络** $q_\phi(z|x)$,一次前向就把任意 $x$ 映成它的近似倾斜——学的是"$x\mapsto$ 倾斜"这个**函数**,连没见过的 $x$ 也能瞬间出近似后验。这是"直接代入 $Qe^g$"给不了的。

### §7.2 VIB 与 ELBO 长得像:是巧合还是同源?

**裁决:不是巧合(coincidence),是被同一个根逼出来的汇流(convergence)。**

**你对的那一半——表面推导确实没"套用恒等式":** VIB 的推法是 ① 写下 IB 目标 $I(Z;X)-\beta I(Z;Y)$;② 把每个 MI 写成 KL,各塞一个变分件(边际塞 $r(z)$、decoder 塞 $q(y|z)$);③ 用 $\mathrm{KL}\ge0$ 夹成界。全程不需要先知道 Gibbs 公式。

**要修正的那一半——为什么不是巧合:**

1. **同一块乐高。** 那两个夹界用的招——"把算不出的真实分布换成一个变分代理,代价正好是个 $\ge0$ 的 KL"——就是 KL 凸对偶性的初等形态,也是 ELBO gap 用的同一招。ELBO 用一块(换后验),VIB 用两块(换边际 + 换 decoder)。同一块乐高搭出同形结构,是**必然**不是运气。
2. **精确区分(诚实加分):** VIB 用的是 **Barber–Agakov / 变分边际界**,**不是**字面的 Donsker–Varadhan 公式——字面套 DV 的是 **MINE**($I=\sup_T\mathbb{E}_{p(x,y)}[T]-\log\mathbb{E}_{p(x)p(y)}[e^T]$)。但 Poole et al. 2019(*On Variational Bounds of MI*)证明 DV/MINE、NWJ、Barber–Agakov、InfoNCE 是**同一个变分-MI 家族、同一个根**:VIB 取"换条件分布"的成员,MINE 取"换 critic 函数"的成员。
3. **决定性一击:** IB 根本不用夹界也能精确解,而它的精确解就是 Gibbs 倾斜 $p(z|x)\propto p(z)e^{-\beta d}$(见 §7.3)。**在做任何近似之前,恒等式就已经坐在 IB 精确最优解的位置上。** 所以不是"近似时碰巧像",是"问题核心本就是它"。

比喻:配方法和求根公式给同一组根,你不会说"恰巧一样"——同一套代数的两种走法。VIB 与 ELBO 也是:同一条 KL 凸对偶,一个从"推断后验"进、一个从"夹 MI"进,落点同形是被代数逼出来的,不是押韵。

### §7.3 恒等式左边 $\log\mathbb{E}_Q e^g$ 在经典 IB 里是谁?

**答案:它就是 $\log Z(x,\beta)$——IB 在输入 $x$ 处能榨出的最优"净值",一个自由能;取平均就是 IB 信息曲线。**

经典 IB 最优 encoder(Tishby 自洽解,$\min I(Z;X)-\beta I(Z;Y)$ 约定):

$$p^\star(z|x)=\frac{p(z)}{Z(x,\beta)}\,e^{-\beta\,d(x,z)},\qquad d(x,z)=\mathrm{KL}\big(p(y|x)\,\|\,p(y|z)\big).$$

> **自洽提醒**:$g=-\beta d$ 里含 $p(y|z)$,而它又依赖 encoder 本身 → 这是个**不动点**,靠 Blahut–Arimoto 迭代解;但每一步的形态都是 $Qe^g$。

把 IB 拉格朗日拆成逐 $x$ 的内层(固定自洽的 $p(z),p(y|z)$):

$$F(x)=\min_P\Big\{\underbrace{\mathrm{KL}(P\|p(z))}_{\text{rate}}+\beta\,\underbrace{\mathbb{E}_P[d(x,z)]}_{\text{distortion}}\Big\}=-\log\underbrace{\mathbb{E}_{p(z)}\big[e^{-\beta d(x,z)}\big]}_{Z(x,\beta)}.$$

(这正是恒等式取负的一侧,$g=-\beta d$、$Q=p(z)$。)于是

$$\boxed{\ \log\mathbb{E}_Q[e^g]=\log Z(x,\beta)=\max_P\{\beta\cdot\text{relevance}-\text{rate}\}\ \text{在输入 }x\text{ 处}.\ }$$

- **逐 $x$**:$\log Z(x,\beta)$ = 该输入能达到的最优"$\beta\times$相关性 − 比特率"净值(一个自由能)。
- **取平均**:$\mathbb{E}_x[\log Z(x,\beta)]=-\mathcal{L}^\star_{IB}-\beta\,I(X;Y)$ = 取负的最优 IB 代价(差一个与 encoder 无关的常数 $\beta I(X;Y)$)= **IB 信息曲线在乘子 $\beta$ 处的那个点**。

这与 ELBO 的 LHS 精确对应:ELBO 里 LHS $=\log p(x)$(log-evidence),IB 里 LHS $=\log Z(x,\beta)$(最优自由能)。两边都是那个**算不出的天花板**,ELBO / VIB 是你能真往上爬的下界。

> **方向校准**:VIB 整体目标是 IB 真值的**下界**(relevance 取下界、rate 取上界,合起来对"要 max 的 IB 值"是下界),正如 ELBO $\le\log p(x)$。"rate 子项取上界"只是内部一步,别和整体方向混了。

### §7.4 总表:恒等式 ↔ ELBO ↔ IB/VIB ↔ Sinkhorn OT

| 恒等式零件 | 通用 | **ELBO / VAE** | **经典 IB / VIB** | **Sinkhorn / 熵正则 OT** |
|---|---|---|---|---|
| 参考 $Q$ | $Q$ | 先验 $p(z)$ | 边际/先验 $p(z)$、$r(z)$ | 独立耦合 $ab^\top$ |
| 打分 $g$ | $g$ | $\log p(x\mid z)$ | $-\beta\,d(x,z)=-\beta\,\mathrm{KL}(p(y\mid x)\,\|\,p(y\mid z))$ | $-C/\varepsilon$(负成本/温度) |
| **左边 $\log\mathbb{E}_Q e^g=\log Z$** | log-配分函数 | **log-evidence $\log p(x)$** | **$\log Z(x,\beta)$ = 最优自由能;均值 $=-\mathcal{L}^\star_{IB}$ = IB 信息曲线** | **对偶 log-sum-exp 软势;最优值 = 熵正则 OT 成本** |
| 最优倾斜 $P^\star\propto Qe^g$ | exponential tilt | **后验 $p(z\mid x)$** | **最优 IB encoder $p(z\mid x)\propto p(z)e^{-\beta d}$(自洽不动点)** | **Gibbs 核耦合 $\operatorname{diag}(u)\,K\,\operatorname{diag}(v),\ K=e^{-C/\varepsilon}$** |
| 变分目标(可跑) | $\mathbb{E}_P[g]-\mathrm{KL}(P\|Q)$ | **ELBO** | **VIB 目标**(rate 上界 + relevance 下界) | **Sinkhorn 迭代**(对偶块坐标上升) |
| 关系 | 目标 $\le$ 左边 | ELBO $\le\log p(x)$ | VIB 目标 $\le\log Z(x,\beta)$ | 等式约束,迭代收敛到**精确**最优(非上下界) |
| gap | $\mathrm{KL}(P\|P^\star)$ | $\mathrm{KL}(q\|p(z\mid x))$ | rate gap $\mathrm{KL}(p(z)\|r)$ + decoder gap $\mathbb{E}_z\mathrm{KL}(p(y\mid z)\|q)$ | 熵正则 vs 真 OT 偏差 $=O(\varepsilon)$,温度 $\varepsilon\to0$ 消失 |

> **OT 与前三列的唯一结构差异**:归一化约束从 **1 个**(总和 $=1$ → 单一配分函数 $Z$)变成 **2 个**(两个边际 → 两个势 $u,v$),所以没有闭式、要 Sinkhorn 轮流归一化。其余(Gibbs 倾斜、自由能、log-配分、Fenchel 对偶、温度极限)完全同源。详见 §7.6。

### §7.5 物理收口:自由能、配分函数、逆温度

$\log Z(x,\beta)$ 字面就是一个**(负)自由能**:它把"能量"(distortion $\beta d$)和"熵"(rate $=\mathrm{KL}$)打包成一个数,$\beta$ 扮演**逆温度**。
- ELBO 世界:真自由能 $=-\log p(x)$,变分自由能 $=-\text{ELBO}$;
- IB 世界:真自由能 $=-\log Z(x,\beta)$,变分自由能 $=-(\text{VIB 目标})$;**扫 $\beta$ = 扫温度**,画出来的"自由能 vs 温度"曲线,就是 §1.5 那条 IB 信息曲线 / rate-distortion 前沿。

**一句话:** 恒等式左边 = "这个问题最好能值多少"。ELBO 里它叫 log-evidence $\log p(x)$;IB 里它叫 $\log Z(x,\beta)$(最优的 $\beta\cdot$相关性 $-$ 比特率,平均起来即 IB 信息曲线)。两边都是那个算不出的天花板,而 ELBO / VIB 是你能真正往上爬的**下界**。

---

### §7.6 Gibbs 核与最优传输:同一台机器,约束从 1 变 2

**这到底是什么:** Sinkhorn 里的 **Gibbs 核** $K=e^{-C/\varepsilon}$,就是 Boltzmann 因子 $e^{-\text{能量}/\text{温度}}$——成本当能量、正则系数 $\varepsilon$ 当温度。熵正则 OT 的最优方案还是指数倾斜 $Qe^g$,只是这次带了**两个**边际约束。

**熵正则最优传输(Cuturi 2013):** 给两个边际 $a,b$、成本矩阵 $C$,求传输方案 $P$(行和 $=a$、列和 $=b$):

$$\min_{P\in U(a,b)}\ \langle P,C\rangle-\varepsilon\,H(P)\ \ \Longleftrightarrow\ \ \min_P\ \langle P,C\rangle+\varepsilon\,\mathrm{KL}(P\,\|\,ab^\top).$$

最优解 $P^\star=\operatorname{diag}(u)\,K\,\operatorname{diag}(v)$,其中 $K_{ij}=e^{-C_{ij}/\varepsilon}$ 就是 **Gibbs 核**,$u,v$ 由 Sinkhorn 迭代 $u\leftarrow a/(Kv),\ v\leftarrow b/(K^\top u)$ 得到。

**逐条接回前文:**

1. **Gibbs 核 = 指数倾斜,目标 = 自由能最小化。** $\langle P,C\rangle-\varepsilon H(P)$ 就是 $\langle\text{能量}\rangle-T\cdot\text{熵}$(§7.5 的自由能),无约束极小是 Gibbs 分布 $\propto e^{-C/\varepsilon}=e^{g}$,参考 $Q=ab^\top$。
2. **唯一区别:约束 1→2。** ELBO 只有 1 个归一化约束(总和 $=1$)→ 单一配分函数 $Z$ → 一锤子 $Qe^g/Z$;OT 有 2 个边际约束 → 两个势 $u,v$ → 没有闭式 → **Sinkhorn 轮流归一化**(一步凑行和 $a$、一步凑列和 $b$,每步就是"算一次配分函数再除")。
3. **对偶又是 log-sum-exp / Fenchel。** 对偶目标含 $\varepsilon\sum_{ij}e^{(f_i+g_j-C_{ij})/\varepsilon}$(软 min = log-配分),primal(熵)↔ dual(log-sum-exp)正是 §1.1 的 Fenchel 共轭;Sinkhorn = 在这条光滑对偶上做**块坐标上升**。
4. **$\varepsilon$ = 温度。** $\varepsilon\to0$:Gibbs 核冻结到最小成本项 → 收敛到真(无正则)OT(= §7.5 两能级 $T\to0$ 冻进基态);$\varepsilon\to\infty$:$K\to$ 全 1 → 独立耦合 $ab^\top$(最大熵)。
5. **信息几何。** entropic OT = 对 Gibbs 参考做 KL 投影(静态 Schrödinger bridge);Sinkhorn = IPF(迭代比例拟合)= 交替 KL / I-投影(Csiszár),和 mirror descent 的 Bregman 投影同源(§1.7)。

**带数字的小例子。** $C=\begin{pmatrix}0&1\\1&0\end{pmatrix}$(对角便宜),$\varepsilon=1$,$a=b=(0.5,0.5)$。Gibbs 核 $K=e^{-C}=\begin{pmatrix}1&0.368\\0.368&1\end{pmatrix}$。解得

$$P^\star=\begin{pmatrix}0.366&0.134\\0.134&0.366\end{pmatrix}\qquad(\text{行/列和}=0.5\ \checkmark).$$

质量压在便宜的对角,$0.134$ 是温度 $\varepsilon=1$ 允许的"**热泄漏**"。$\varepsilon\to0$ → $\begin{pmatrix}0.5&0\\0&0.5\end{pmatrix}$(hard OT,成本 0);$\varepsilon\to\infty$ → $\begin{pmatrix}0.25&0.25\\0.25&0.25\end{pmatrix}$(独立耦合,最大熵)。

**aha:** Gibbs 核就是 Gibbs 分布在"双边际约束"下的化身,Sinkhorn 是它的迭代解法。**Gibbs 分布 → Gibbs 自由能 → ELBO/VIB → 经典 IB 最优 encoder → Sinkhorn 的 Gibbs 核**,全是"$Q$ 乘 $e^{\text{打分}}$ 再归一化 + 用配分/对偶记账"这一个母体,在不同约束、不同 $(Q,g,T)$ 下的化身。

**tie-back:** 见 §7.4 总表新增的 OT 列;Gibbs 核 = 该列的"最优倾斜",$\varepsilon$ 对应 IB 的温度 $1/\beta$,Sinkhorn 的交替投影对应 §1.7 的 Bregman / mirror-descent 几何。

---

## §8 收尾主线:从一个恒等式,一路走到 VAE

> 这一节把前面十几节拧成**一条主线**——只读这节就能建立全局,细节随时跳回 §1 / §7。一句话剧透:**全程只有一个动作(「$Q$ 乘 $e^{\text{打分}}$ 再归一化」)和一句记账(「得分 = 天花板 − 到完美答案的距离」),其余都是在不同约束、不同 $(Q,g)$、不同工程封装下给它换名字。**

### 一颗种子

$$\log\mathbb{E}_Q[e^{g}]=\sup_{P}\big\{\mathbb{E}_P[g]-\mathrm{KL}(P\|Q)\big\},\qquad P^\star\propto Q\,e^{g}.$$

- 左边 = **log-配分函数** = 你能榨出的最大净值 = 物理上的**负自由能**;
- 右边 = 在「追分 $\mathbb{E}_P[g]$」与「别离参考 $Q$ 太远的过路费 $\mathrm{KL}$」之间拔河;
- 赢家永远是**指数倾斜** $P^\star\propto Qe^g$(§1.2 硬币:倾斜=sigmoid,$\log Z=\log 2.5=0.916$;§7.5:这就是 Boltzmann 分布,$Q$ 当参考、$g=-E/T$)。

### 一条主线(七个车站)

**① 恒等式 → 自由能。** 代入 $g=-E/T$,左边就是 $-T\ln Z$ = (Helmholtz/Gibbs)自由能,$\sup$ 就是「能量 vs 熵」的拔河(§7.5)。**配分函数 = 负自由能 = 我们一路在榨的「果汁」。**

**② 恒等式 → 「得分 = 天花板 − gap」(Donsker–Varadhan 对偶)。** 固定一个 $P$(= 你的近似 $q$),恒等式给出
$$\mathbb{E}_q[g]-\mathrm{KL}(q\|Q)=\underbrace{\log\mathbb{E}_Q[e^g]}_{\text{天花板(定值)}}-\underbrace{\mathrm{KL}(q\|P^\star)}_{\text{到完美答案 }P^\star\text{ 的距离}}.$$
**最大化看得见的得分 ⟺ 缩小看不见的、到 $P^\star$ 的距离。** 海拔计比喻:山顶高度(天花板)和山顶位置($P^\star$)都看不见,但海拔计(得分)每涨一分,你离山顶就近一分。(维基「Duality Formula for Variational Inference」就是这条,只是它的 $P/Q$ 字母与本文对调:它的参考叫 $P$、变量叫 $Q$。)

**③ 实例化成贝叶斯。** 取参考 $Q=$ 先验 $p(z)$、打分 $g=\log p(x|z)$,于是:倾斜 $P^\star\propto p(z)p(x|z)=$ **后验 $p(z|x)$**;天花板 = **log-evidence $\log p(x)$**;得分 = **ELBO**;gap = $\mathrm{KL}(q\|p(z|x))$。即 $\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q\|p(z|x))$(§2 ①)。

**④ 命门:为什么不能直接拿 $P^\star$。** $P^\star\propto Qe^g=p(x,z)$ 只是**形状**(= 未归一化后验,**点点可算**:$p(x,z)=p(z)p(x|z)$ 是我们自己写的公式,代数字即得,如 $\log p(2,1)=-2.84$,§7.1);缺的归一化常数 $=p(x)=\int p(x,z)\,dz=Z$ 是**积不出的高维积分**——这才是命门。**ELBO 的天才:$\mathcal L(q)=\mathbb{E}_q[\log p(x,z)]-\mathbb{E}_q[\log q(z)]$ 只用点点可算的 $p(x,z)$ 和我们自选的 $q$(能算能采样),彻底绕开那个积分。**

**⑤ VI = 在能算的族里爬 ELBO。** 把②那个 $\sup$ 的范围,从「所有分布」缩到「一个 tractable 族 $\mathcal Q$」(mean-field / 高斯 / 神经网络)。无限制版给精确后验(但够不着 $Z$);**受限版给族内最优近似——这就是 variational inference**。族装得下真后验 → gap=0 精确命中(高斯-高斯例);族弯不到 → reverse-KL 让 $q$ 抱核心、压方差(mean-field 拟合相关高斯,方差 $1\to0.19$)。

**⑥ VI → VAE:三个工程动作。**
- **A 摊销 (amortize)**:别对每个 $x$ 单独优化,训**一个 encoder 网络** $q_\phi(z|x)$,一次前向出 $(\mu_\phi(x),\sigma_\phi(x))$(它学的是「$x\mapsto$ 后验参数」这条函数,如线性高斯里的 $0.5x$)。代价:amortization gap。
- **B reparameterize**:$z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\ \epsilon\sim N(0,I)$,把采样变成对 $\phi$ 可导,从而能 SGD。
- **C 神经解码器 + 联合学**:$p_\theta(x|z)$ 做成网络,$\theta,\phi$ **一起** SGD 爬 ELBO。
合体即 **VAE**:$\max_{\theta,\phi}\sum_i\big[\mathbb{E}_{q_\phi(z|x_i)}[\log p_\theta(x_i|z)]-\mathrm{KL}(q_\phi(z|x_i)\|p(z))\big]$,读作「autoencoder + 把码拉向 $N(0,I)$ 的 KL」,而它逐字就是 ELBO。**白送彩蛋:生成**——$z\sim N(0,I)\to$ 解码出新样本。

**⑦ 同一台机器的其他化身。** 换 $(Q,g,T)$ 或换约束数:经典 IB 最优 encoder $\propto p(z)e^{-\beta d}$(§7.3)、β-VAE(扫 $\beta$ = 扫温度)、Sinkhorn 的 Gibbs 核(双边际约束,§7.6)、mirror descent(KL=Bregman,§1.7)、RLHF-KL 最优策略……全在 §7.4 总表那几列里。

### 主线图

```
            一颗种子:  log E_Q[e^g] = sup_P { E_P[g] − KL(P‖Q) },   P* ∝ Q e^g
                                          │
      ┌───────────────────────────────────┼───────────────────────────────────┐
      ▼                                   ▼                                     ▼
① 自由能视角                      ② 得分 = 天花板 − gap (DV)              ④ 命门:p(x,z) 点点可算,
 −T ln Z,能量 vs 熵              max 得分 ⟺ min 到 P* 的 KL              但 ∫p(x,z)dz = p(x) = Z 算不出
                                          │                                     │
                                          ▼                                     │
                                 ③ 贝叶斯实例化                                 │
                          Q=先验, g=log p(x|z) ⇒ P*=后验,                     │
                          天花板=log p(x), 得分=ELBO, gap=KL(q‖后验)           │
                                          │   ←──── ELBO 只用 p(x,z)+q,绕开 Z ──┘
                                          ▼
                                 ⑤ VI = 在能算的族里爬 ELBO
                                          │
                       A 摊销(encoder) + B reparameterize + C 神经 decoder 联合学
                                          ▼
                                 ⑥ VAE(并白送生成 z~N(0,I)→解码)
                                          │
        ⑦ 换 (Q,g,T) / 换约束 → 经典IB·VIB · β-VAE · Sinkhorn Gibbs核 · mirror descent · RLHF-KL
```

### 主线一句话

> **认领一个参考 $Q$、一个打分 $g$,「乘 $e^g$ 再归一化」就给出最优倾斜 $P^\star$,而你能榨出的净值就是 log-配分函数(= 负自由能 = 天花板)。当 $P^\star$ 因归一化积分 $Z$ 算不出时,就退而在一个能算的族里爬「得分 = 天花板 − gap」的下界(ELBO)——这是 VI;再把推断摊销成 encoder、用 reparameterization 变可微、把解码器也一起学——这是 VAE。从 Gibbs 到 VAE,自始至终是同一台机器。**

---

## 一句话收尾

整份报告(和这份伴读)其实只在反复讲**一道菜谱**:**拿一个参考分布 $Q$,逐点乘 $e^{\text{打分}}$,再归一化**。换食材($X/Y/\beta$/先验/打分)就换出 sigmoid、softmax、Boltzmann、后验、ELBO、VIB、SOE 的逐维 rate 门限。你感到的那套"迷人的统一数学",底层就是这一下 + 它的两张对偶地图(log-配分函数 ↔ KL)。
