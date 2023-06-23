
def do_rounds(algo,turns,mode):
    regrets = []
    algo.set_exp_eta()
    if mode == 2:
        algo.precompute_semi_bandit()
    for i in range(turns):
        algo.regenerate_cost()
        choice,index = algo.make_a_choice()
        loss = algo.get_loss(choice)
        regret = algo.dynamic_regret(choice)
        regrets.append(regret)
        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            algo.run_bandit(loss,choice)


        algo.exp2_main()

    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/(i+1))
    return over_time_val
def do_rounds_osmd(algo,turns,mode,):
    regrets = []
    algo.osmd_pre()
    algo.set_osmd_eta()
    if mode == 2:
        algo.precompute_semi_bandit()
    for i in range(turns):
        algo.regenerate_cost()
        algo.set_osmd_pt()
        choice,index = algo.make_a_choice()

        regret = algo.dynamic_regret(choice)
        regrets.append(regret)

        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            loss = algo.get_loss(choice)
            algo.run_bandit(loss,choice)

        algo.run_osmd()

    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/(i+1))
    return over_time_val
def do_rounds_random(algo,turns,mode):
    regrets = []
    if mode == 2:
        algo.precompute_semi_bandit()
    for i in range(turns):
        algo.regenerate_cost()
        choice,index = algo.make_a_choice()
        regret = algo.dynamic_regret(choice)
        regrets.append(regret)
        if mode == 2:
            algo.run_semi_bandit(choice)
        elif mode ==3:
            loss = algo.get_loss(choice)
            algo.run_bandit(loss,choice)

    over_time_val=[]
    sum_val=0
    for i in range(len(regrets)):
        sum_val+=regrets[i]
        over_time_val.append(sum_val/(i+1))
    return over_time_val

