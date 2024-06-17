""" 
Description: 
 - Utility for selectai. This util execute selectai SQL directly,
   for example:
   select ai runsql 'who am i'

History:
 - 2024/04/29 by Hysun (hysun.he@oracle.com): Initial version
"""

import config
from typing import List, Optional
from loguru import logger


def runsql(sentence: str, llm_profile: str) -> Optional[List[any]]:
    result_array = []
    sql = f"select ai runsql '{sentence}'"
    with config.selectai_pool.acquire() as connection:
        with connection.cursor() as cursor:
            cursor.callproc(
                "dbms_cloud_ai.set_profile",
                [llm_profile],
            )
            cursor.execute(sql)
            cols = cursor.description
            data = cursor.fetchall()
            for row in data:
                logger.info(f"*** Row: {row} with cols {cols}")
                if "could not be generated for you" in str(row[0]):
                    return None
                row_json = dict()
                result_array.append(row_json)
                for c in range(len(row)):
                    row_json[cols[c][0]] = row[c]

    logger.info(f"### Result JSON(runsql): {result_array}")
    return result_array


def chat(sentence: str, llm_profile: str) -> Optional[str]:
    sql = f"select ai chat '{sentence}'"
    with config.selectai_pool.acquire() as connection:
        with connection.cursor() as cursor:
            cursor.callproc(
                "dbms_cloud_ai.set_profile",
                [llm_profile],
            )
            for result in cursor.execute(sql):
                logger.info(f"*** FreeChat Output ***")
                logger.info(result[0])
                logger.info(f"***********************")
                return result[0]
    return None  # Expectation is that here is unreachable.
