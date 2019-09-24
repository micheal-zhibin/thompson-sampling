## usage

``` javascript
    import bandit from 'thompson-sampling';
    /**
     * function k_arm_bandit
     * @params 
     * [Array]      ctrs        k个臂预设的收益率
     * [Number]     alpha       容错概率
     * [Number]     burn_in     熔断次数
     * [Number]     burn_in     熔断次数
     * [Number]     max_iter    最大迭代次数
     * [Number]     draw        每次预估尝试次数
     * [Boolean]    slient      是否展示熔断日志
     */
    const result = bandit.k_arm_bandit(ctrs, alpha, burn_in, max_iter, draw, slient);
```