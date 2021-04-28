import sys
from os.path import join
sys.path.insert(0, join('..', '..'))
from utils.root_finder import find_root
import lark

from datasets.comp_gen_bundle import install_sql_datasets, Registry
install_sql_datasets()

print(f"all_names: {' '.join(Registry._datasets.keys())}")
# for dname, fn in Registry._datasets.items():
#     train, dev, test = fn()
#     for example in iter(train):
#         pass

def parse_sql(ds_name='atis_iid'):
    train, dev, test = Registry.get_dataset(ds_name)
    print(f"{ds_name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    example = next(iter(test))
    print(example['sql'])

    # from ebnf_compiler import pretty_repr_tree, pretty_derivation_tree
    lark_text = join(find_root(), 'src', 'statics', 'grammar', 'SQLite.lark')
    parser = lark.Lark(open(lark_text), start="parse", keep_all_tokens=True, )
    # lark_text = join(find_root(), 'src', 'statics', 'grammar', 'MySQL.lark')
    # parser = lark.Lark(open(lark_text), start="query", keep_all_tokens=True, )
    from itertools import chain
    for i, example in enumerate(chain(iter(train), iter(dev), iter(test))):
        sql = example['sql']
        try:
            tree = parser.parse(sql)
            print(f"OK {i}")
        except KeyboardInterrupt:
            return
        except:
            print(sql)
        # print(pretty_repr_tree(tree))
        # print('\n'.join(pretty_derivation_tree(tree)))

def main():
    names = filter(lambda n: 'iid' in n, Registry._datasets.keys())
    for name in names:
        print('-' * 30)
        print(name)
        parse_sql(name)

def inspect():
    from sklearn import decomposition
    # ds_name = 'atis_iid'
    # train, dev, test = Registry.get_dataset(ds_name)
    # print(f"{ds_name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
    # sql0 = train[0]['sql']
    # sql1 = train[927]['sql']
    #
    # ds_name = 'geo_iid'
    # train, dev, test = Registry.get_dataset(ds_name)
    # print(f"{ds_name}: train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
    # sql2 = train[44]['sql']
    # print(sql2)
    # return

    sql0 = """
    SELECT DISTINCT FLIGHTalias0.FLIGHT_ID
    FROM AIRPORT AS AIRPORTalias0 , AIRPORT_SERVICE AS AIRPORT_SERVICEalias0 , CITY AS CITYalias0 ,
    DATE_DAY AS DATE_DAYalias0 , DAYS AS DAYSalias0 , FLIGHT AS FLIGHTalias0
    WHERE ( ( CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = "city_name0" AND DATE_DAYalias0.DAY_NUMBER = "day_number0"
    AND DATE_DAYalias0.MONTH_NUMBER = month_number0 AND DATE_DAYalias0.YEAR = year0 AND DAYSalias0.DAY_NAME = DATE_DAYalias0.DAY_NAME
    AND FLIGHTalias0.FLIGHT_DAYS = DAYSalias0.DAYS_CODE AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE )
    AND AIRPORTalias0.AIRPORT_CODE = "airport_code0" AND FLIGHTalias0.FROM_AIRPORT = AIRPORTalias0.AIRPORT_CODE ) AND FLIGHTalias0.STOPS = "stops0"
    """
    sql1 = """
    SELECT DISTINCT AIRCRAFTalias0.AIRCRAFT_CODE
    FROM AIRCRAFT AS AIRCRAFTalias0
    WHERE AIRCRAFTalias0.CAPACITY = (
        SELECT MIN( AIRCRAFTalias1.CAPACITY )
        FROM AIRCRAFT AS AIRCRAFTalias1
        WHERE AIRCRAFTalias1.CAPACITY > ALL (
            SELECT DISTINCT AIRCRAFTalias2.CAPACITY FROM AIRCRAFT AS AIRCRAFTalias2
            WHERE AIRCRAFTalias1.PROPULSION = "propulsion0"
        )
    ) AND AIRCRAFTalias0.CAPACITY > ALL (
        SELECT DISTINCT AIRCRAFTalias3.CAPACITY FROM AIRCRAFT AS AIRCRAFTalias3
        WHERE AIRCRAFTalias3.PROPULSION = "propulsion0"
    ) ;
    """
    sql2 = """
    SELECT COUNT( RIVERalias0.RIVER_NAME ) FROM RIVER AS RIVERalias0
    WHERE RIVERalias0.LENGTH > ALL (
        SELECT RIVERalias1.LENGTH FROM RIVER AS RIVERalias1 WHERE RIVERalias1.RIVER_NAME = "river_name0"
    ) AND RIVERalias0.TRAVERSE = "state_name0" ;
    """
    sql3 = """
    SELECT DISTINCT COUNT( PAPERalias0.PAPERID ) FROM AUTHOR AS AUTHORalias0 , PAPER AS PAPERalias0 , WRITES AS WRITESalias0 WHERE AUTHORalias0.AUTHORNAME = "authorname0" AND PAPERalias0.YEAR == YEAR(CURDATE()) - misc0 AND WRITESalias0.AUTHORID = AUTHORalias0.AUTHORID AND WRITESalias0.PAPERID = PAPERalias0.PAPERID ;
    """

    from ebnf_compiler import pretty_repr_tree, pretty_derivation_tree
    # lark_text = join(find_root(), 'src', 'statics', 'grammar', 'SQLite.lark')
    # parser = lark.Lark(open(lark_text), start="parse", keep_all_tokens=True)
    lark_text = join(find_root(), 'src', 'statics', 'grammar', 'MySQL.lark')
    parser = lark.Lark(open(lark_text), start="query", keep_all_tokens=True, )
    try:
        sql = sql3
        print(sql)
        tree = parser.parse(sql)
        print(pretty_repr_tree(tree))
        print('\n'.join(pretty_derivation_tree(tree)))
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    inspect()
    # main()
    # ATIS iid: idx: 927 (train)
    # Geo iid: idx: 44 (train)
    # advising:
    # scholar:
