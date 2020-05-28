#!/bin/bash
set -e 

data_dir="F:\徐佳明量子物理\matlab_data"

# 整理数据
bash search.sh $data_dir png > data.lst # linux

cat data.lst | awk -F "/" '{split($(NF-1),A,"_");printf("%s %s %s %s\n",$0,$(NF-1),A[1],A[2])}' > data.label
cat data.label | awk 'BEGIN{i1=-1;label1[null]="";i2=-1;label2[null]="";i3=-1;label3[null]=""};
			{if(!($(NF-2) in label1)){label1[$(NF-2)]++;i1++};
			if(!($(NF-1) in label2)){label2[$(NF-1)]++;i2++}; 
				if(!($NF in label3)){label3[$NF]++;i3++};
			printf("%s %s %s %s\n",$1,i1,i2,i3);}' > data.label.num

# 划分 训练/开发/测试 集合
cat data.label.num | shuf -n 1000 > data.dev
cat data.label.num | shuf -n 1000 > data.test
cat data.label.num | grep -v -f data.dev -f data.test > data.train

# 训练 
