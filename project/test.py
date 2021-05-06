def MCT_Nex(player,opt,state,choice,close_kong=0):
    """
    状态转移函数
    """
    if opt==0 :
        if state==0 :
            return player,3,0
        if state==1 :
            return player,3,4
    if opt==1 :
        if state==0 :
            return player,0,0
        if state==1 :
            return (player+1)%4,0,0
        if state==2 :
            return player,1,0
        if state==3 :
            return player,3,7
        if state==4 :
            return player,0,1
        if state==5:
            return player,2,0
    if opt==2 :
        if state==0 :
            return player,3,1
        if state==1 :
            return player,3,2
        if state==2 :
            return player,3,5
        if state==3 :
            return player,3,6
    if opt==3 :
        if state==0 :
            if choice==0 :
                return player,4,0
            else :
                return player,8,0
        if state==1 :
            if choice==0 :
                return player,4,1
            else :
                return player,8,0
        if state==2 :
            if choice==0 :
                return player,4,1
            else :
                return player,8,0
        if state==3 :
            if choice==0 :
                return player,0,1
            else :
                return player,8,0
        if state==4 :
            if choice==0 :
                return player,4,2
            else :
                return player,8,0
        if state==5 :
            if choice==0 :
                return player,4,3
            else :
                return player,8,0
        if state==6 :
            if choice==0 :
                return player,4,3
            else :
                return player,8,0
        if state==7 :
            if choice==0 :
                return player,0,1
            else :
                return player,8,0
    if opt==4 :
        if state==0:
            if choice==0 :
                return player,7,0
            else :
                if close_kong==1:
                    return player,1,0
                else :
                    return player,3,3
        if state==1 :
            if choice==0 :
                return player,5,0
            else :
                return (player+choice-1)%4,0,1
        if state==2:
            if choice==0 :
                return player,7,1
            else :
                if close_kong==1:
                    return player,1,2
                else :
                    return player,1,3
        if state==3:
            if choice==0 :
                return player,5,1
            else :
                return (player+choice-1)%4,1,4
    if opt==5 :
        if state==0:
            if choice==0 :
                return player,6,0
            else :
                return (player+(choice-1)//4)%4,2,0
        if state==1:
            if choice==0 :
                return player,6,1
            else :
                return (player+(choice-1)//4)%4,1,5
    if opt==6 :
        if state==0:
            if choice==0 :
                return (player+1)%4,0,0
            else :
                return (player+1)%4,2,0
        if state==1:
            if choice==0 :
                return player,1,1
            else :
                return (player+1)%4,1,5
    if opt==7 :
        if state==0 :
            if choice==0 :
                return player,2,0
            else :
                return player,2,1
        if state==1 :
            if choice==0 :
                return player,2,2
            else :
                return player,2,3
    return -1,-1,-1
x=MCT_Nex(0,0,0,0,0)
print(x)