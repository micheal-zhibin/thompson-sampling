const PD = require('probability-distributions');

class Arm {
    constructor(idex, a=1, b=1) {
        this.idex = idex;
        this.a = a;
        this.b = b;
    }

    record_success() {
        this.a ++;
    }

    record_fail() {
        this.b ++;
    }

    draw_ctr() {
        return PD.rbeta(1, this.a, this.b)[0];
    }

    mean() {
        return this.a / (this.a + this.b);
    }
}

function thompson_sampling(arms) {
    const all_sample = arms.map((arm) => {
        return {
            idex: arm.idex,
            ctr: arm.draw_ctr()
        };
    });
    return all_sample.sort((a,b)=>{return b.ctr-a.ctr})[0];
}

function monte_carlo_simulation(arms, draw = 100) {
    const alphas = [];
    const betas = [];
    const mc = [];
    const winner_idxs = [];

    arms.forEach((arm) => {
        alphas.push(arm.a);
        betas.push(arm.b);
    });

    for (let i = 0; i < draw; i ++) {
        const temp = [];
        arms.forEach((arm) => {
            temp.push(arm.draw_ctr());
        });
        mc.push(temp);
        winner_idxs.push(temp.indexOf(Math.max(...temp)));
    }

    // console.log('winner_idxs', winner_idxs)

    const counts = [...Array(arms.length)].map(_=>0);

    winner_idxs.forEach((item) => {
        counts[item] ++;
    })

    const p_winner = counts.map((count) => {
        return count / draw;
    });

    // console.log('p_winner', p_winner)

    return {
        mc,
        p_winner
    };
}

function should_terminate(p_winner, est_ctrs, mc, alpha=0.05) {
    const winner_idx = p_winner.indexOf(Math.max(...p_winner));
    const values_remaining = mc.map((item) => {
        const max = Math.max(...item);
        return (max - item[winner_idx]) / item[winner_idx];
    });
    const q = 1 - alpha;
    const position = values_remaining.length * q;
    if (Math.round(position) === position) {
        return values_remaining.sort()[position] < 0.01 * est_ctrs[winner_idx];
    } else {
        const temp = values_remaining.sort();
        return (temp[Math.floor(position)] + temp[Math.ceil(position)]) / 2 < 0.01 * est_ctrs[winner_idx];
    }
}

function k_arm_bandit(ctrs, alpha=0.05, burn_in=1000, max_iter=100, draw=100, slient=false) {
    const n_arms = ctrs.length;
    const arms = [];
    let est_ctrs, idx;
    const history_p = [];
    for (let i = 0; i < n_arms; i ++) {
        arms.push(new Arm(i));
        history_p.push([]);
    }

    let i = 0;

    for (; i < max_iter; i ++) {
        // console.log('---i---', i);
        idx = thompson_sampling(arms).idex;
        // debugger
        const arm = arms[idx];
        const ctr = ctrs[idx];

        if (Math.random() < ctr) {
            arm.record_success();
        } else {
            arm.record_fail();
        }

        const { mc, p_winner } = monte_carlo_simulation(arms, draw);
        // console.log('p_winner', p_winner)
        for (let j = 0; j < p_winner.length; j ++) {
            // console.log(history_p[j])
            history_p[j].push(p_winner[j]);
        }

        est_ctrs = arms.map((arm) => {
            return arm.mean();
        });

        console.log('should_terminate(p_winner, est_ctrs, mc, alpha)', should_terminate(p_winner, est_ctrs, mc, alpha))

        if (i >= burn_in && should_terminate(p_winner, est_ctrs, mc, alpha)) {
            if (!slient) {
                console.log(`Terminated at iteration ${i}`);
            }
            break;
        }
    }
    const traffic = arms.map((arm) => {
        console.log('arm.a', arm.a, 'arm.b', arm.b);
        return arm.a + arm.b - 2;
    });
    return {
        idx,
        i,
        est_ctrs,
        history_p,
        traffic
    }
}

export default {
    Arm,
    thompson_sampling,
    monte_carlo_simulation,
    should_terminate,
    k_arm_bandit
};