import argparse
import crt_binary_table
import crt_abstract_episodes

parser = argparse.ArgumentParser()
parser.add_argument('--autor',type=str)
parser.add_argument('--values_agt',type=bool,default=False)
parser.add_argument('--values_trnng',type=bool,default=False)
parser.add_argument('--combine',type=bool,default=False )

args = parser.parse_args()

crt_abstract_episodes.args = args 
# create_abstract_episodes(
#         args.autor,
#         create_episodes_agt=args.values_agt,
#         create_episodes_trnng=args.values_trnng
#         )


crt_binary_table.args = args
crt_binary_table.crt_binary_table(
        autor=args.autor,
        values_agt=args.values_agt, 
        values_trnng=args.values_agt, 
    )

