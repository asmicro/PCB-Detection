import pandas as pd
def create_df(path="cache/detect/labels/1.txt"):
    res=[]
    classes = ["missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"]
    dict_out={}
    for i in classes:
        dict_out[i]=0
    with open(path) as f:
        out=[int(line[0]) for line in f.readlines()]
    for i in out:
        dict_out[classes[i]]+=1
    for j,i in  enumerate(list(dict_out.keys())):                    
        res.append({'Object': i, 'S.No': j+1, 'Quantity': dict_out[i]})
    # print(res)
    df = pd.DataFrame.from_records(res, index='S.No')
    return(df)

if __name__=="__main__":
    print(create_df())
        

