import sys
sys.path.insert(0, '../..')
import shujuji.cg_bundle as cg_bundle
from os.path import join

def _get_parsed_sql_ds(data_name: str, *, use_iid: bool, grammar_file: str, startpoint: str = None):
    ds_dir = join(cg_bundle.CG_DATA_PATH, data_name, 'new_question_split' if use_iid else 'schema_full_split')

    def _build_ds(filename: str):
        nonlocal ds_dir
        ds = cg_bundle.LarkParserDatasetWrapper(
            grammar_filename=grammar_file,
            startpoint=startpoint if startpoint is not None else ('parse' if 'sqlite' in grammar_file.lower() else 'query'),
            parse_keys=['sql'],
            dataset=cg_bundle.FlattenSeqDS(cg_bundle.JsonDataset(join(ds_dir, filename)), sql_only=True)
        )
        return ds

    train = _build_ds('aligned_train.json')
    dev = _build_ds('aligned_final_dev.json')
    test = _build_ds('final_test.json')
    print(f"load dataset: {ds_dir}")
    return train, dev, test

def main(ds_name: str = 'scholar'):
    print('-----' * 10)
    print(ds_name)
    print('-----' * 10)
    train, dev, test = _get_parsed_sql_ds(
        ds_name,
        # use_iid=False, grammar_file=join('run', 'scholar_cg.sqlite.20.lark'), startpoint='parse'
        use_iid = False, grammar_file = '../../statics/grammar/SQLite.lark', startpoint = 'parse'
    )
    parser = train.parser
    sql = """
    SELECT DISTINCT FLIGHTalias0.FLIGHT_ID
    FROM TABLE_PLACEHOLDER AS AIRPORT_SERVICEalias0 , TABLE_PLACEHOLDER AS AIRPORT_SERVICEalias1 ,
         TABLE_PLACEHOLDER AS CITYalias0 , TABLE_PLACEHOLDER AS CITYalias1 , TABLE_PLACEHOLDER AS DATE_DAYalias0 ,
         TABLE_PLACEHOLDER AS DATE_DAYalias1 , TABLE_PLACEHOLDER AS DAYSalias0 , TABLE_PLACEHOLDER AS DAYSalias1 ,
         TABLE_PLACEHOLDER AS FAREalias0 , TABLE_PLACEHOLDER AS FAREalias1 , TABLE_PLACEHOLDER AS FARE_BASISalias0 ,
         TABLE_PLACEHOLDER AS FARE_BASISalias1 , TABLE_PLACEHOLDER AS FARE_BASISalias2 , TABLE_PLACEHOLDER AS FLIGHTalias0 ,
         TABLE_PLACEHOLDER AS FLIGHT_FAREalias0 , TABLE_PLACEHOLDER AS FLIGHT_FAREalias1
    WHERE (
        ( ( ( ( ( ( (
                      ( FLIGHTalias0.ARRIVAL_TIME < arrival_time0 OR FLIGHTalias0.TIME_ELAPSED >= time_elapsed0 )
                      AND FLIGHTalias0.DEPARTURE_TIME > FLIGHTalias0.ARRIVAL_TIME
                    )
                    AND DATE_DAYalias1.DAY_NUMBER = day_number0 AND DATE_DAYalias1.MONTH_NUMBER = month_number0
                    AND DATE_DAYalias1.YEAR = year0 AND DAYSalias1.DAY_NAME = DATE_DAYalias1.DAY_NAME
                    AND FLIGHTalias0.FLIGHT_DAYS = DAYSalias1.DAYS_CODE
                  )
                  AND DATE_DAYalias0.DAY_NUMBER = day_number0 AND DATE_DAYalias0.MONTH_NUMBER = month_number0
                  AND DATE_DAYalias0.YEAR = year0 AND DAYSalias0.DAY_NAME = DATE_DAYalias0.DAY_NAME
                  AND FARE_BASISalias0.CLASS_TYPE = 'class_type0' AND FARE_BASISalias1.BASIS_DAYS = DAYSalias0.DAYS_CODE
                  AND FAREalias0.FARE_BASIS_CODE = FARE_BASISalias0.FARE_BASIS_CODE
                  AND FAREalias0.FARE_BASIS_CODE = FARE_BASISalias1.FARE_BASIS_CODE
                  AND FLIGHT_FAREalias0.FARE_ID = FAREalias0.FARE_ID AND FLIGHTalias0.FLIGHT_ID = FLIGHT_FAREalias0.FLIGHT_ID
                )
                OR
                (
                  (
                    DATE_DAYalias1.DAY_NUMBER = day_number1 AND DATE_DAYalias1.MONTH_NUMBER = month_number0
                    AND DATE_DAYalias1.YEAR = year0 AND DAYSalias1.DAY_NAME = DATE_DAYalias1.DAY_NAME
                    AND FLIGHTalias0.FLIGHT_DAYS = DAYSalias1.DAYS_CODE
                    AND NOT (
                       ( FLIGHTalias0.ARRIVAL_TIME < arrival_time0 OR FLIGHTalias0.TIME_ELAPSED >= time_elapsed0 )
                       AND
                       FLIGHTalias0.DEPARTURE_TIME > FLIGHTalias0.ARRIVAL_TIME
                    )
                  )
                  AND DATE_DAYalias0.DAY_NUMBER = day_number1 AND DATE_DAYalias0.MONTH_NUMBER = month_number0
                  AND DATE_DAYalias0.YEAR = year0 AND DAYSalias0.DAY_NAME = DATE_DAYalias0.DAY_NAME
                  AND FARE_BASISalias0.CLASS_TYPE = 'class_type0' AND FARE_BASISalias1.BASIS_DAYS = DAYSalias0.DAYS_CODE
                  AND FAREalias0.FARE_BASIS_CODE = FARE_BASISalias0.FARE_BASIS_CODE
                  AND FAREalias0.FARE_BASIS_CODE = FARE_BASISalias1.FARE_BASIS_CODE
                  AND FLIGHT_FAREalias0.FARE_ID = FAREalias0.FARE_ID
                  AND FLIGHTalias0.FLIGHT_ID = FLIGHT_FAREalias0.FLIGHT_ID
                )
              )
              AND FARE_BASISalias2.CLASS_TYPE = 'class_type0' AND FAREalias1.FARE_BASIS_CODE = FARE_BASISalias2.FARE_BASIS_CODE
              AND FLIGHT_FAREalias1.FARE_ID = FAREalias1.FARE_ID AND FLIGHTalias0.FLIGHT_ID = FLIGHT_FAREalias1.FLIGHT_ID
            )
            AND FLIGHTalias0.ARRIVAL_TIME < arrival_time1
          )
          AND CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND CITYalias1.CITY_NAME = 'city_name0'
          AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE
        )
        AND CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = 'city_name1'
        AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE
    )
    AND FLIGHTalias0.AIRLINE_CODE = 'airline_code0' ;
    """
    # sql = test.dataset[29]['sql']
    sql = """
    SELECT DISTINCT PAPERalias0.JOURNALID  ,  PAPERalias0.YEAR
    FROM TABLE_PLACEHOLDER AS AUTHORalias0  ,  TABLE_PLACEHOLDER AS PAPERalias0  ,  TABLE_PLACEHOLDER AS WRITESalias0
    WHERE AUTHORalias0.AUTHORNAME = 'authorname0'
        AND WRITESalias0.AUTHORID = AUTHORalias0.AUTHORID
        AND WRITESalias0.PAPERID = PAPERalias0.PAPERID
    GROUP BY PAPERalias0.JOURNALID  ,  PAPERalias0.YEAR ORDER BY PAPERalias0.YEAR DESC ;
    """
    print(sql)
    print(parser.parse(sql).pretty())

    return
    def _ds_stat(dataset):
        none_num = total = 0
        for i, x in enumerate(dataset):
            if x['sql_tree'] is None:
                none_num += 1
                print(f"{i}th example failed to parse.")
            total += 1

        print(f"Failed to parse: {none_num} / {total} = {none_num / total}")

    _ds_stat(train)
    _ds_stat(dev)
    _ds_stat(test)

if __name__ == '__main__':
    # main('geography')
    main('scholar')
    # main('atis')
    # main('advising')
