## 使用 vs 探索
`使用 vs 探索` 困境在我们生活中多方面存在着。就好像是，你喜欢的饭店在路的右边角落里。如果你每天都去那里，你会对你想要的很满意，但是却会减少了寻找更好饭馆的机会。如果你总想着尝试新的东西，那么你很可能每次都吃到不太满意的食物。同样的，线上一些广告投放者尝试在已知的最好的收益的广告和新的可能有更好的广告之间找到更好的平衡。

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_117f3ec3-5fdf-43ad-8dfd-eef7ea6e3f8f.png)

如果我们把环境相关的所有信息都知晓了，即使是用的简单的 `BF` 算法我们也可以找到最好策略，获得不错的提升。刚才提到的问题就是因为我们没有办法获取全部信息：我们需要获取足够的信息再去在所有的决定中找出最优解同时控制着风险。通过使用，我们可以从我们最优解中获知优势。通过探索，我们可以利用风险去收集我们一些未知的选择相关的信息。最好的长期策略需要一些短期的牺牲。例如，一个探索方案有可能是一个大错误，但是它却提醒着我们不要再将来经常采取同一形式。

## 什么是多臂 Bandit 问题？
[多臂 bandit 问题](https://en.wikipedia.org/wiki/Multi-armed_bandit)是一个用来证明使用 vs 探索困境的完美经典问题。想象你在一家赌场里，面前是多台老虎机并且每台机子的收益概率都未知的。这个时候你面对的问题是：用哪种投注策略可以获得最大的长期收益？

在这种场景下，我们唯一需要去讨论的是怎样可以进行无限次的试验完成目的。限定次数的试验又是描述另一种新的探索问题。举个栗子，如果实验次数小于老虎机台数，我们没办法把每台机器的回报率确定下来并因此我们要利用好我们很少的知识和资源并更加聪明去做决策。

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_d1604232-f69d-4246-acd7-96393bb9e08a.png)

一个更加现实的形容是你持续去玩同一台机器很多很多次你最终可以根据大数定律确认到机器的收益率。但是，这会造成浪费并且也不能保证能得出最好的长期收益。

### 定义
接下来让我给大家声明一些科学的定义。

伯努利多臂 bandit 问题可以被描述为一组⟨A, R⟩，具体如下：
- 我们有 K 台机器还有对应的回报率{θ1,…,θK}
- 在每一次的时间步骤 t，我们都会找一台老虎机实践并且获得收益 r
- A 是一组实践，每一个都关联上一台老虎机。每个实践的值就是一个期待的收益，Q(a)=E[r|a] = θ。如果实践at是在时间步骤t是在第i台机器上，然后 Q(at)=θi
- R 是一个收益函数。在伯努利 bandit 问题上，我们假设收益r是一个随机概率。在时间步骤t时，rt=R(at)会在概率Q(at)情况下返回1否则返回0

这就是一个简单的马克洛夫决策过程，这里说的是一个没有状态的 S

我们的目标是找到一个最大化的累计回报∑(T,t=1)rt。如果我们已知最优解的最理想操作，那么我们的目标就会和最小化选择非最优解时潜在遗憾或者
失败一致。

最优解 a* 的最优回报概率 θ* 可以表示为：
![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_b9559bd4-73e1-4649-897f-6dc9e3648ec0.png)

遗憾函数就是我们在时间步骤 T 时没有选到最优解时的总遗憾：
![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_53824e13-e529-44ab-9e1e-607b2c4ece26.png)

## Bandit 策略
下面是基于我们怎么取做探索的一些解决多臂 bandit 问题方法
- 不探索：最原始并且最差的一种
- 随机探索
- 对于未知更偏向于聪明的去探索

## ε-贪婪算法
ε-贪婪算法在大部分情况下表现都是很好的，但是在随机探索情况下偶现异常。行为值是根据已经观察了很久（到目前时间步骤 t）目标 行为 a 的历史回报来估计的：

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_765f829d-47dc-4ba5-9272-a6c442f77998.png)

这里 `𝟙` 是一个二进制指示函数，Nt(a)是行为已经被总共选择了多少次，所以 ![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_dcb674e7-c772-405d-ac64-e59ed135f2e9.png)

根据 ε-贪婪算法，在小的概率 ϵ 下我们执行一个随机决策，但是其他情况下（大部分情况下，概率是 1-ϵ）我们选择我们已经验证过的最好的决策：![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_4f5b4557-8d30-452e-8763-36e35bb8c730.png)

(点击可以查看我的小栗子)[https://github.com/lilianweng/multi-armed-bandit/blob/master/solvers.py#L45]

## 置信区间上界
随机探索给我们机会去尝试我们所有未知的选择。但是，由于随机性，我们会在一个已经在之前确定不好的决策中停下来。为了避免这种低效的探索，一个做法是及时减少参数 ε 同时其其他高度位置的策略保持乐观并因此选择我们还不能百分百确定收益预估的策略。换而言之，我们更喜欢探索有比较大可能性获得更好的收益的策略。

UCB 算法通过计算收益的置信区间上界 Û t(a) 来确定策略的潜力，所以真实值很大概率上是低于边界值 Q(a)≤Q̂ t(a)+Û t(a)。上界Û t(a)是 Nt(a)的函数，一个大量实践集合 Nt(a)会让我们得到更小的边界值Û t(a)

在 UCB 算法中，我们总是选择最贪心的策略来最大化满足更高的置信区间上界：
![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_6d498c5b-946d-4295-92a5-0c69ce0e8bc8.png)

现在，我们要面对的问题是怎么去估算这个置信区间上界

### 霍夫丁不等式
如果我们不想确定任何关于分布展示的前置知识，我们可以从[“Hoeffding’s Inequality”](http://cs229.stanford.edu/extra-notes/hoeffding.pdf)--一个适合任何边界分布的理论获得不少的提示。

假设 X1, ... , Xt 是独立同分布的随机变量，他们都在区间[0, 1]中。样本标识为![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_07c68917-dd84-48e5-b885-8778f173e4bf.png)。对于u > 0，我们能得到：

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_eef08ee8-1e7e-472e-8bf6-a251e347511a.png)

给一个指定的策略 a，我们可以认为：
- rt(a) 是随机变量
- Q(a) 作为真平均数
- Q̂ t(a)作为样本平均数
- u 为置信区间上界，u=Ut(a)

然后我们可以得出，
![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_e124b638-4e80-405f-aa95-f869a9cb99c6.png)

我们想要去选择一个边区间可以在大概率上让真平均数小于样本平均数+置信区间上界。因此 e^−2tUt(a)2 应该是一个小概率，我们可以认为一个很窄的门槛 p 是符合的

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_655adcb2-9fe0-4acd-9a02-2d0a035c602e.png)

### UCB1
另外一个启发去及时减少门槛 p 的是我们想用更多的回报观察去估算更多的置信区间。让我们把 p=t^-4 时，我们可以得到 UCB1 算法：

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_1d3e4b39-7323-40d2-94d8-51fc225bfaee.png)

### 贝叶斯 UCB
在 UCB 或者 UCB1 算法中，我们不去假设任何基于回报分布的前置假设并且我们不得不依赖霍夫丁不等式去得出每个估算。如果我们可以知道直接知道分布，我们可以做更好的区间预测。

举个栗子，如果我们期待每个老虎机的平均收益符合图二中的高斯分布，我们可以通过把Û t(a)
设为标准差的两倍使得上界被认为95%可信。

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_f6b817bc-e786-47fb-b8e0-2293ebd0ae74.png)

## 汤普森采样
汤普森采样虽然是一个简单的想法但是却能在解决多臂 bandit 问题中发挥着很大的作用。

在每一个时间步骤中，我们想要选择的策略 a 的概率来确定策略 a 是最优的：

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_bb8e8f2c-d7de-4316-808d-0de5d61459e4.png)

π(a|ht)是基于历史ht选择策略 a 的概率

在伯努利 bandit 中，我们可以自然假设 Q(a) 服从 Beta 分布，同时 Q(a) 的成功概率 θ 也符合伯努利分布。Beta(α,β) 的值在区间 [0, 1] 内，α 和 β 分别对应我们成功或失败获得回报。

首先，让我们基于一些前置历史或者每个策略的主题初始化 Beta 参数 α 和 β。举个简单栗子，
- α = 1 和 β = 1；我们期望回报率为50%同时我们不太自信
- α = 1000 和 β = 9000；我们对回报率达到10%比较有信息

在每一个时间t，我们可以从每一个策略的历史Beta(αi,βi) 抽样一个预计回报 Q̃ (a)。在这些样本中找到最好的策略： aTSt=argmaxa∈AQ̃ (a)。在真实回报率统计出来后，我们可以根据这个来更新我们的 Beta 分布，本质上通过贝叶斯推断把已知历史和相似获得的样本信息算出后续的回报概率。

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_f928e7cd-385b-40c4-b031-0ba7333ca7b6.png)

汤普森采样是基于概率匹配的实现。因为它的回报预估 Q̃ 是后续分布的样本，这里每种概率都等效于对应在之前实验过程中最优解的概率。

但是，在一些更实际更复杂的问题上，需要通过更加棘手的计算才能通过贝叶斯推断和观察到的真实回报把预估的后续分布推算出来。汤普森采样在我们可以通过如Gibbs sampling, Laplace approximate, 和 bootstraps 方法预估后续分布的时候发挥其作用。这个[指引](https://arxiv.org/pdf/1707.02038.pdf)有一个比较好的整体回顾；如果你想了解更多关于汤普森采样相关的知识，我强烈推荐这个指引。

## 实例学习
我在[lilianweng/multi-armed-bandit](https://github.com/lilianweng/multi-armed-bandit)实现了一个小栗子。一个[BernoulliBandit](https://github.com/lilianweng/multi-armed-bandit/blob/master/bandits.py#L13)对象由一系列随机或预定义回报概率组成。bandit 算法被实现为[Solver](https://github.com/lilianweng/multi-armed-bandit/blob/master/solvers.py#L9)的子类，并且用一个 bandit 对象最为目标问题。累计的遗憾都被及时跟踪。

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_873b7eb5-c27d-4ee6-8ec3-7fcea2d0caf1.png)

## 小结
我们需要探索，因为信息是宝贵的。在探索策略方面，我们可以完全不做探索，只关注与短期收益。或者我们偶尔的随机去探索。在未来，我们探索并且有选择兴趣的挑选方案进行探索--越不确定策略会被喜欢因为他们可以获取到更丰富的信息。

![](https://michealfeng-1257331036.cos.ap-guangzhou.myqcloud.com/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_0a3a98e6-af1d-4a6c-9eaf-02ab8b00623e.png)