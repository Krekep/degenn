import gspread


def get_cell_dict_for_sheet(
    title: str,
) -> dict[
    tuple[str, str, str], dict[str, dict[str, dict[str, dict[str, gspread.cell.Cell]]]]
]:
    curr_sheet = table.worksheet(title)
    all_cells = curr_sheet.get_all_cells()
    cells_dict = {
        (
            f"{chr(64 + cell.col)}{cell.row}"
            if cell.col <= 26
            else (
                f"{chr(64 + cell.col // 26)}{chr(64 + cell.col % 26)}{cell.row}"
                if cell.col != 52
                else f"AZ{cell.row}"
            )
        ): cell
        for cell in all_cells
    }

    # key --- algorithm description --- name, parameters
    block_cells = {
        ("Random"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["AP6"],
                        "Count": cells_dict["AP7"],
                        "Launch 20": cells_dict["AP9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AO6"],
                        "Count": cells_dict["AO7"],
                        "Launch 20": cells_dict["AO9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AN6"],
                        "Count": cells_dict["AN7"],
                        "Launch 20": cells_dict["AN9"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["AS6"],
                        "Count": cells_dict["AS7"],
                        "Launch 20": cells_dict["AS9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AR6"],
                        "Count": cells_dict["AR7"],
                        "Launch 20": cells_dict["AR9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AQ6"],
                        "Count": cells_dict["AQ7"],
                        "Launch 20": cells_dict["AQ9"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AV6"],
                        "Count": cells_dict["AV7"],
                        "Launch 20": cells_dict["AV9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AU6"],
                        "Count": cells_dict["AU7"],
                        "Launch 20": cells_dict["AU9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AT6"],
                        "Count": cells_dict["AT7"],
                        "Launch 20": cells_dict["AT9"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["AY6"],
                        "Count": cells_dict["AY7"],
                        "Launch 20": cells_dict["AY9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AX6"],
                        "Count": cells_dict["AX7"],
                        "Launch 20": cells_dict["AX9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AW6"],
                        "Count": cells_dict["AW7"],
                        "Launch 20": cells_dict["AW9"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["BB6"],
                        "Count": cells_dict["BB7"],
                        "Launch 20": cells_dict["BB9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["BA6"],
                        "Count": cells_dict["BA7"],
                        "Launch 20": cells_dict["BA9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AZ6"],
                        "Count": cells_dict["AZ7"],
                        "Launch 20": cells_dict["AZ9"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["BE6"],
                        "Count": cells_dict["BE7"],
                        "Launch 20": cells_dict["BE9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["BD6"],
                        "Count": cells_dict["BD7"],
                        "Launch 20": cells_dict["BD9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["BC6"],
                        "Count": cells_dict["BC7"],
                        "Launch 20": cells_dict["BC9"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["AP16"],
                        "Count": cells_dict["AP17"],
                        "Launch 20": cells_dict["AP19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AO16"],
                        "Count": cells_dict["AO17"],
                        "Launch 20": cells_dict["AO19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AN16"],
                        "Count": cells_dict["AN17"],
                        "Launch 20": cells_dict["AN19"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["AS16"],
                        "Count": cells_dict["AS17"],
                        "Launch 20": cells_dict["AS19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AR16"],
                        "Count": cells_dict["AR17"],
                        "Launch 20": cells_dict["AR19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AQ16"],
                        "Count": cells_dict["AQ17"],
                        "Launch 20": cells_dict["AQ19"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AV16"],
                        "Count": cells_dict["AV17"],
                        "Launch 20": cells_dict["AV19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AU16"],
                        "Count": cells_dict["AU17"],
                        "Launch 20": cells_dict["AU19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AT16"],
                        "Count": cells_dict["AT17"],
                        "Launch 20": cells_dict["AT19"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["AY16"],
                        "Count": cells_dict["AY17"],
                        "Launch 20": cells_dict["AY19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AX16"],
                        "Count": cells_dict["AX17"],
                        "Launch 20": cells_dict["AX19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AW16"],
                        "Count": cells_dict["AW17"],
                        "Launch 20": cells_dict["AW19"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["BB16"],
                        "Count": cells_dict["BB17"],
                        "Launch 20": cells_dict["BB19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["BA16"],
                        "Count": cells_dict["BA17"],
                        "Launch 20": cells_dict["BA19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AZ16"],
                        "Count": cells_dict["AZ17"],
                        "Launch 20": cells_dict["AZ19"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["BE16"],
                        "Count": cells_dict["BE17"],
                        "Launch 20": cells_dict["BE19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["BD16"],
                        "Count": cells_dict["BD17"],
                        "Launch 20": cells_dict["BD19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["BC16"],
                        "Count": cells_dict["BC17"],
                        "Launch 20": cells_dict["BC19"],
                    },
                },
            },
        },
        ("Ann", "lin", "dc[300]"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["D6"],
                        "Count": cells_dict["D7"],
                        "Launch 20": cells_dict["D9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["C6"],
                        "Count": cells_dict["C7"],
                        "Launch 20": cells_dict["C9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["B6"],
                        "Count": cells_dict["B7"],
                        "Launch 20": cells_dict["B9"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["G6"],
                        "Count": cells_dict["G7"],
                        "Launch 20": cells_dict["G9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["F6"],
                        "Count": cells_dict["F7"],
                        "Launch 20": cells_dict["F9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["E6"],
                        "Count": cells_dict["E7"],
                        "Launch 20": cells_dict["E9"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["J6"],
                        "Count": cells_dict["J7"],
                        "Launch 20": cells_dict["J9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["I6"],
                        "Count": cells_dict["I7"],
                        "Launch 20": cells_dict["I9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["H6"],
                        "Count": cells_dict["H7"],
                        "Launch 20": cells_dict["H9"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["M6"],
                        "Count": cells_dict["M7"],
                        "Launch 20": cells_dict["M9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["L6"],
                        "Count": cells_dict["L7"],
                        "Launch 20": cells_dict["L9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["K6"],
                        "Count": cells_dict["K7"],
                        "Launch 20": cells_dict["K9"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["P6"],
                        "Count": cells_dict["P7"],
                        "Launch 20": cells_dict["P9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["O6"],
                        "Count": cells_dict["O7"],
                        "Launch 20": cells_dict["O9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["N6"],
                        "Count": cells_dict["N7"],
                        "Launch 20": cells_dict["N9"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["S6"],
                        "Count": cells_dict["S7"],
                        "Launch 20": cells_dict["S9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["R6"],
                        "Count": cells_dict["R7"],
                        "Launch 20": cells_dict["R9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["Q6"],
                        "Count": cells_dict["Q7"],
                        "Launch 20": cells_dict["Q9"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["D16"],
                        "Count": cells_dict["D17"],
                        "Launch 20": cells_dict["D19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["C16"],
                        "Count": cells_dict["C17"],
                        "Launch 20": cells_dict["C19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["B16"],
                        "Count": cells_dict["B17"],
                        "Launch 20": cells_dict["B19"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["G16"],
                        "Count": cells_dict["G17"],
                        "Launch 20": cells_dict["G19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["F16"],
                        "Count": cells_dict["F17"],
                        "Launch 20": cells_dict["F19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["E16"],
                        "Count": cells_dict["E17"],
                        "Launch 20": cells_dict["E19"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["J16"],
                        "Count": cells_dict["J17"],
                        "Launch 20": cells_dict["J19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["I16"],
                        "Count": cells_dict["I17"],
                        "Launch 20": cells_dict["I19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["H16"],
                        "Count": cells_dict["H17"],
                        "Launch 20": cells_dict["H19"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["M16"],
                        "Count": cells_dict["M17"],
                        "Launch 20": cells_dict["M19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["L16"],
                        "Count": cells_dict["L17"],
                        "Launch 20": cells_dict["L19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["K16"],
                        "Count": cells_dict["K17"],
                        "Launch 20": cells_dict["K19"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["P16"],
                        "Count": cells_dict["P17"],
                        "Launch 20": cells_dict["P19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["O16"],
                        "Count": cells_dict["O17"],
                        "Launch 20": cells_dict["O19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["N16"],
                        "Count": cells_dict["N17"],
                        "Launch 20": cells_dict["N19"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["S16"],
                        "Count": cells_dict["S17"],
                        "Launch 20": cells_dict["S19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["R16"],
                        "Count": cells_dict["R17"],
                        "Launch 20": cells_dict["R19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["Q16"],
                        "Count": cells_dict["Q17"],
                        "Launch 20": cells_dict["Q19"],
                    },
                },
            },
        },
        ("Ann", "exp", "dc[300]"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["W6"],
                        "Count": cells_dict["W7"],
                        "Launch 20": cells_dict["W9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["V6"],
                        "Count": cells_dict["V7"],
                        "Launch 20": cells_dict["V9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["U6"],
                        "Count": cells_dict["U7"],
                        "Launch 20": cells_dict["U9"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["Z6"],
                        "Count": cells_dict["Z7"],
                        "Launch 20": cells_dict["Z9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["Y6"],
                        "Count": cells_dict["Y7"],
                        "Launch 20": cells_dict["Y9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["X6"],
                        "Count": cells_dict["X7"],
                        "Launch 20": cells_dict["X9"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AC6"],
                        "Count": cells_dict["AC7"],
                        "Launch 20": cells_dict["AC9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AB6"],
                        "Count": cells_dict["AB7"],
                        "Launch 20": cells_dict["AB9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AA6"],
                        "Count": cells_dict["AA7"],
                        "Launch 20": cells_dict["AA9"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["AF6"],
                        "Count": cells_dict["AF7"],
                        "Launch 20": cells_dict["AF9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AE6"],
                        "Count": cells_dict["AE7"],
                        "Launch 20": cells_dict["AE9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AD6"],
                        "Count": cells_dict["AD7"],
                        "Launch 20": cells_dict["AD9"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["AI6"],
                        "Count": cells_dict["AI7"],
                        "Launch 20": cells_dict["AI9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AH6"],
                        "Count": cells_dict["AH7"],
                        "Launch 20": cells_dict["AH9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AG6"],
                        "Count": cells_dict["AG7"],
                        "Launch 20": cells_dict["AG9"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AL6"],
                        "Count": cells_dict["AL7"],
                        "Launch 20": cells_dict["AL9"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AK6"],
                        "Count": cells_dict["AK7"],
                        "Launch 20": cells_dict["AK9"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AJ6"],
                        "Count": cells_dict["AJ7"],
                        "Launch 20": cells_dict["AJ9"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["W16"],
                        "Count": cells_dict["W17"],
                        "Launch 20": cells_dict["W19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["V16"],
                        "Count": cells_dict["V17"],
                        "Launch 20": cells_dict["V19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["U16"],
                        "Count": cells_dict["U17"],
                        "Launch 20": cells_dict["U19"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["Z16"],
                        "Count": cells_dict["Z17"],
                        "Launch 20": cells_dict["Z19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["Y16"],
                        "Count": cells_dict["Y17"],
                        "Launch 20": cells_dict["Y19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["X16"],
                        "Count": cells_dict["X17"],
                        "Launch 20": cells_dict["X19"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AC16"],
                        "Count": cells_dict["AC17"],
                        "Launch 20": cells_dict["AC19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AB16"],
                        "Count": cells_dict["AB17"],
                        "Launch 20": cells_dict["AB19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AA16"],
                        "Count": cells_dict["AA17"],
                        "Launch 20": cells_dict["AA19"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["AF16"],
                        "Count": cells_dict["AF17"],
                        "Launch 20": cells_dict["AF19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AE16"],
                        "Count": cells_dict["AE17"],
                        "Launch 20": cells_dict["AE19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AD16"],
                        "Count": cells_dict["AD17"],
                        "Launch 20": cells_dict["AD19"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["AI16"],
                        "Count": cells_dict["AI17"],
                        "Launch 20": cells_dict["AI19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AH16"],
                        "Count": cells_dict["AH17"],
                        "Launch 20": cells_dict["AH19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AG16"],
                        "Count": cells_dict["AG17"],
                        "Launch 20": cells_dict["AG19"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AL16"],
                        "Count": cells_dict["AL17"],
                        "Launch 20": cells_dict["AL19"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AK16"],
                        "Count": cells_dict["AK17"],
                        "Launch 20": cells_dict["AK19"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AJ16"],
                        "Count": cells_dict["AJ17"],
                        "Launch 20": cells_dict["AJ19"],
                    },
                },
            },
        },
        ("Ann", "lin", "dc[400]"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["D26"],
                        "Count": cells_dict["D27"],
                        "Launch 20": cells_dict["D29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["C26"],
                        "Count": cells_dict["C27"],
                        "Launch 20": cells_dict["C29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["B26"],
                        "Count": cells_dict["B27"],
                        "Launch 20": cells_dict["B29"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["G26"],
                        "Count": cells_dict["G27"],
                        "Launch 20": cells_dict["G29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["F26"],
                        "Count": cells_dict["F27"],
                        "Launch 20": cells_dict["F29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["E26"],
                        "Count": cells_dict["E27"],
                        "Launch 20": cells_dict["E29"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["J26"],
                        "Count": cells_dict["J27"],
                        "Launch 20": cells_dict["J29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["I26"],
                        "Count": cells_dict["I27"],
                        "Launch 20": cells_dict["I29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["H26"],
                        "Count": cells_dict["H27"],
                        "Launch 20": cells_dict["H29"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["M26"],
                        "Count": cells_dict["M27"],
                        "Launch 20": cells_dict["M29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["L26"],
                        "Count": cells_dict["L27"],
                        "Launch 20": cells_dict["L29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["K26"],
                        "Count": cells_dict["K27"],
                        "Launch 20": cells_dict["K29"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["P26"],
                        "Count": cells_dict["P27"],
                        "Launch 20": cells_dict["P29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["O26"],
                        "Count": cells_dict["O27"],
                        "Launch 20": cells_dict["O29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["N26"],
                        "Count": cells_dict["N27"],
                        "Launch 20": cells_dict["N29"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["S26"],
                        "Count": cells_dict["S27"],
                        "Launch 20": cells_dict["S29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["R26"],
                        "Count": cells_dict["R27"],
                        "Launch 20": cells_dict["R29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["Q26"],
                        "Count": cells_dict["Q27"],
                        "Launch 20": cells_dict["Q29"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["D36"],
                        "Count": cells_dict["D37"],
                        "Launch 20": cells_dict["D39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["C36"],
                        "Count": cells_dict["C37"],
                        "Launch 20": cells_dict["C39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["B36"],
                        "Count": cells_dict["B37"],
                        "Launch 20": cells_dict["B39"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["G36"],
                        "Count": cells_dict["G37"],
                        "Launch 20": cells_dict["G39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["F36"],
                        "Count": cells_dict["F37"],
                        "Launch 20": cells_dict["F39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["E36"],
                        "Count": cells_dict["E37"],
                        "Launch 20": cells_dict["E39"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["J36"],
                        "Count": cells_dict["J37"],
                        "Launch 20": cells_dict["J39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["I36"],
                        "Count": cells_dict["I37"],
                        "Launch 20": cells_dict["I39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["H36"],
                        "Count": cells_dict["H37"],
                        "Launch 20": cells_dict["H39"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["M36"],
                        "Count": cells_dict["M37"],
                        "Launch 20": cells_dict["M39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["L36"],
                        "Count": cells_dict["L37"],
                        "Launch 20": cells_dict["L39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["K36"],
                        "Count": cells_dict["K37"],
                        "Launch 20": cells_dict["K39"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["P36"],
                        "Count": cells_dict["P37"],
                        "Launch 20": cells_dict["P39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["O36"],
                        "Count": cells_dict["O37"],
                        "Launch 20": cells_dict["O39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["N36"],
                        "Count": cells_dict["N37"],
                        "Launch 20": cells_dict["N39"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["S36"],
                        "Count": cells_dict["S37"],
                        "Launch 20": cells_dict["S39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["R36"],
                        "Count": cells_dict["R37"],
                        "Launch 20": cells_dict["R39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["Q36"],
                        "Count": cells_dict["Q37"],
                        "Launch 20": cells_dict["Q39"],
                    },
                },
            },
        },
        ("Ann", "exp", "dc[400]"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["W26"],
                        "Count": cells_dict["W27"],
                        "Launch 20": cells_dict["W29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["V26"],
                        "Count": cells_dict["V27"],
                        "Launch 20": cells_dict["V29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["U26"],
                        "Count": cells_dict["U27"],
                        "Launch 20": cells_dict["U29"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["Z26"],
                        "Count": cells_dict["Z27"],
                        "Launch 20": cells_dict["Z29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["Y26"],
                        "Count": cells_dict["Y27"],
                        "Launch 20": cells_dict["Y29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["X26"],
                        "Count": cells_dict["X27"],
                        "Launch 20": cells_dict["X29"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AC26"],
                        "Count": cells_dict["AC27"],
                        "Launch 20": cells_dict["AC29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AB26"],
                        "Count": cells_dict["AB27"],
                        "Launch 20": cells_dict["AB29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AA26"],
                        "Count": cells_dict["AA27"],
                        "Launch 20": cells_dict["AA29"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["AF26"],
                        "Count": cells_dict["AF27"],
                        "Launch 20": cells_dict["AF29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AE26"],
                        "Count": cells_dict["AE27"],
                        "Launch 20": cells_dict["AE29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AD26"],
                        "Count": cells_dict["AD27"],
                        "Launch 20": cells_dict["AD29"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["AI26"],
                        "Count": cells_dict["AI27"],
                        "Launch 20": cells_dict["AI29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AH26"],
                        "Count": cells_dict["AH27"],
                        "Launch 20": cells_dict["AH29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AG26"],
                        "Count": cells_dict["AG27"],
                        "Launch 20": cells_dict["AG29"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AL26"],
                        "Count": cells_dict["AL27"],
                        "Launch 20": cells_dict["AL29"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AK26"],
                        "Count": cells_dict["AK27"],
                        "Launch 20": cells_dict["AK29"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AJ26"],
                        "Count": cells_dict["AJ27"],
                        "Launch 20": cells_dict["AJ29"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["W36"],
                        "Count": cells_dict["W37"],
                        "Launch 20": cells_dict["W39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["V36"],
                        "Count": cells_dict["V37"],
                        "Launch 20": cells_dict["V39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["U36"],
                        "Count": cells_dict["U37"],
                        "Launch 20": cells_dict["U39"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["Z36"],
                        "Count": cells_dict["Z37"],
                        "Launch 20": cells_dict["Z39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["Y36"],
                        "Count": cells_dict["Y37"],
                        "Launch 20": cells_dict["Y39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["X36"],
                        "Count": cells_dict["X37"],
                        "Launch 20": cells_dict["X39"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AC36"],
                        "Count": cells_dict["AC37"],
                        "Launch 20": cells_dict["AC39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AB36"],
                        "Count": cells_dict["AB37"],
                        "Launch 20": cells_dict["AB39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AA36"],
                        "Count": cells_dict["AA37"],
                        "Launch 20": cells_dict["AA39"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["AF36"],
                        "Count": cells_dict["AF37"],
                        "Launch 20": cells_dict["AF39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AE36"],
                        "Count": cells_dict["AE37"],
                        "Launch 20": cells_dict["AE39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AD36"],
                        "Count": cells_dict["AD37"],
                        "Launch 20": cells_dict["AD39"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["AI36"],
                        "Count": cells_dict["AI37"],
                        "Launch 20": cells_dict["AI39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AH36"],
                        "Count": cells_dict["AH37"],
                        "Launch 20": cells_dict["AH39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AG36"],
                        "Count": cells_dict["AG37"],
                        "Launch 20": cells_dict["AG39"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AL36"],
                        "Count": cells_dict["AL37"],
                        "Launch 20": cells_dict["AL39"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AK36"],
                        "Count": cells_dict["AK37"],
                        "Launch 20": cells_dict["AK39"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AJ36"],
                        "Count": cells_dict["AJ37"],
                        "Launch 20": cells_dict["AJ39"],
                    },
                },
            },
        },
        ("Ann", "lin", "dl[50][400]"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["D46"],
                        "Count": cells_dict["D47"],
                        "Launch 20": cells_dict["D49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["C46"],
                        "Count": cells_dict["C47"],
                        "Launch 20": cells_dict["C49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["B46"],
                        "Count": cells_dict["B47"],
                        "Launch 20": cells_dict["B49"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["G46"],
                        "Count": cells_dict["G47"],
                        "Launch 20": cells_dict["G49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["F46"],
                        "Count": cells_dict["F47"],
                        "Launch 20": cells_dict["F49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["E46"],
                        "Count": cells_dict["E47"],
                        "Launch 20": cells_dict["E49"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["J46"],
                        "Count": cells_dict["J47"],
                        "Launch 20": cells_dict["J49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["I46"],
                        "Count": cells_dict["I47"],
                        "Launch 20": cells_dict["I49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["H46"],
                        "Count": cells_dict["H47"],
                        "Launch 20": cells_dict["H49"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["M46"],
                        "Count": cells_dict["M47"],
                        "Launch 20": cells_dict["M49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["L46"],
                        "Count": cells_dict["L47"],
                        "Launch 20": cells_dict["L49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["K46"],
                        "Count": cells_dict["K47"],
                        "Launch 20": cells_dict["K49"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["P46"],
                        "Count": cells_dict["P47"],
                        "Launch 20": cells_dict["P49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["O46"],
                        "Count": cells_dict["O47"],
                        "Launch 20": cells_dict["O49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["N46"],
                        "Count": cells_dict["N47"],
                        "Launch 20": cells_dict["N49"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["S46"],
                        "Count": cells_dict["S47"],
                        "Launch 20": cells_dict["S49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["R46"],
                        "Count": cells_dict["R47"],
                        "Launch 20": cells_dict["R49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["Q46"],
                        "Count": cells_dict["Q47"],
                        "Launch 20": cells_dict["Q49"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["D56"],
                        "Count": cells_dict["D57"],
                        "Launch 20": cells_dict["D59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["C56"],
                        "Count": cells_dict["C57"],
                        "Launch 20": cells_dict["C59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["B56"],
                        "Count": cells_dict["B57"],
                        "Launch 20": cells_dict["B59"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["G56"],
                        "Count": cells_dict["G57"],
                        "Launch 20": cells_dict["G59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["F56"],
                        "Count": cells_dict["F57"],
                        "Launch 20": cells_dict["F59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["E56"],
                        "Count": cells_dict["E57"],
                        "Launch 20": cells_dict["E59"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["J56"],
                        "Count": cells_dict["J57"],
                        "Launch 20": cells_dict["J59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["I56"],
                        "Count": cells_dict["I57"],
                        "Launch 20": cells_dict["I59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["H56"],
                        "Count": cells_dict["H57"],
                        "Launch 20": cells_dict["H59"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["M56"],
                        "Count": cells_dict["M57"],
                        "Launch 20": cells_dict["M59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["L56"],
                        "Count": cells_dict["L57"],
                        "Launch 20": cells_dict["L59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["K56"],
                        "Count": cells_dict["K57"],
                        "Launch 20": cells_dict["K59"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["P56"],
                        "Count": cells_dict["P57"],
                        "Launch 20": cells_dict["P59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["O56"],
                        "Count": cells_dict["O57"],
                        "Launch 20": cells_dict["O59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["N56"],
                        "Count": cells_dict["N57"],
                        "Launch 20": cells_dict["N59"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["S56"],
                        "Count": cells_dict["S57"],
                        "Launch 20": cells_dict["S59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["R56"],
                        "Count": cells_dict["R57"],
                        "Launch 20": cells_dict["R59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["Q56"],
                        "Count": cells_dict["Q57"],
                        "Launch 20": cells_dict["Q59"],
                    },
                },
            },
        },
        ("Ann", "exp", "dl[50][400]"): {
            "MeanAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["W46"],
                        "Count": cells_dict["W47"],
                        "Launch 20": cells_dict["W49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["V46"],
                        "Count": cells_dict["V47"],
                        "Launch 20": cells_dict["V49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["U46"],
                        "Count": cells_dict["U47"],
                        "Launch 20": cells_dict["U49"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["Z46"],
                        "Count": cells_dict["Z47"],
                        "Launch 20": cells_dict["Z49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["Y46"],
                        "Count": cells_dict["Y47"],
                        "Launch 20": cells_dict["Y49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["X46"],
                        "Count": cells_dict["X47"],
                        "Launch 20": cells_dict["X49"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AC46"],
                        "Count": cells_dict["AC47"],
                        "Launch 20": cells_dict["AC49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AB46"],
                        "Count": cells_dict["AB47"],
                        "Launch 20": cells_dict["AB49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AA46"],
                        "Count": cells_dict["AA47"],
                        "Launch 20": cells_dict["AA49"],
                    },
                },
            },
            "MaxAbsolutePercentageError": {
                "50": {
                    "50": {
                        "Time(s)": cells_dict["AF46"],
                        "Count": cells_dict["AF47"],
                        "Launch 20": cells_dict["AF49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AE46"],
                        "Count": cells_dict["AE47"],
                        "Launch 20": cells_dict["AE49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AD46"],
                        "Count": cells_dict["AD47"],
                        "Launch 20": cells_dict["AD49"],
                    },
                },
                "25": {
                    "50": {
                        "Time(s)": cells_dict["AI46"],
                        "Count": cells_dict["AI47"],
                        "Launch 20": cells_dict["AI49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AH46"],
                        "Count": cells_dict["AH47"],
                        "Launch 20": cells_dict["AH49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AG46"],
                        "Count": cells_dict["AG47"],
                        "Launch 20": cells_dict["AG49"],
                    },
                },
                "10": {
                    "50": {
                        "Time(s)": cells_dict["AL46"],
                        "Count": cells_dict["AL47"],
                        "Launch 20": cells_dict["AL49"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AK46"],
                        "Count": cells_dict["AK47"],
                        "Launch 20": cells_dict["AK49"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AJ46"],
                        "Count": cells_dict["AJ47"],
                        "Launch 20": cells_dict["AJ49"],
                    },
                },
            },
            "MaxAbsoluteDeviation": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["W56"],
                        "Count": cells_dict["W57"],
                        "Launch 20": cells_dict["W59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["V56"],
                        "Count": cells_dict["V57"],
                        "Launch 20": cells_dict["V59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["U56"],
                        "Count": cells_dict["U57"],
                        "Launch 20": cells_dict["U59"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["Z56"],
                        "Count": cells_dict["Z57"],
                        "Launch 20": cells_dict["Z59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["Y56"],
                        "Count": cells_dict["Y57"],
                        "Launch 20": cells_dict["Y59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["X56"],
                        "Count": cells_dict["X57"],
                        "Launch 20": cells_dict["X59"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AC56"],
                        "Count": cells_dict["AC57"],
                        "Launch 20": cells_dict["AC59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AB56"],
                        "Count": cells_dict["AB57"],
                        "Launch 20": cells_dict["AB59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AA56"],
                        "Count": cells_dict["AA57"],
                        "Launch 20": cells_dict["AA59"],
                    },
                },
            },
            "RootMeanSquaredError": {
                "3": {
                    "50": {
                        "Time(s)": cells_dict["AF56"],
                        "Count": cells_dict["AF57"],
                        "Launch 20": cells_dict["AF59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AE56"],
                        "Count": cells_dict["AE57"],
                        "Launch 20": cells_dict["AE59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AD56"],
                        "Count": cells_dict["AD57"],
                        "Launch 20": cells_dict["AD59"],
                    },
                },
                "1": {
                    "50": {
                        "Time(s)": cells_dict["AI56"],
                        "Count": cells_dict["AI57"],
                        "Launch 20": cells_dict["AI59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AH56"],
                        "Count": cells_dict["AH57"],
                        "Launch 20": cells_dict["AH59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AG56"],
                        "Count": cells_dict["AG57"],
                        "Launch 20": cells_dict["AG59"],
                    },
                },
                "0.1": {
                    "50": {
                        "Time(s)": cells_dict["AL56"],
                        "Count": cells_dict["AL57"],
                        "Launch 20": cells_dict["AL59"],
                    },
                    "150": {
                        "Time(s)": cells_dict["AK56"],
                        "Count": cells_dict["AK57"],
                        "Launch 20": cells_dict["AK59"],
                    },
                    "400": {
                        "Time(s)": cells_dict["AJ56"],
                        "Count": cells_dict["AJ57"],
                        "Launch 20": cells_dict["AJ59"],
                    },
                },
            },
        },
    }
    return block_cells


gc = gspread.service_account()
table = gc.open_by_key("1Rs8ZiGKfYV5SJpzC1cMe_q2773BesMl51v7lKyypSQA")
work_sheets = table.worksheets()


sheets = {
    "lin": get_cell_dict_for_sheet("Lin"),
    "exp": get_cell_dict_for_sheet("Exp"),
    "sin": get_cell_dict_for_sheet("Sin"),
    "log": get_cell_dict_for_sheet("Log"),
    "sig": get_cell_dict_for_sheet("Sig"),
    "gauss": get_cell_dict_for_sheet("Gauss"),
    "hyperbol": get_cell_dict_for_sheet("Hyperbol"),
    "const": get_cell_dict_for_sheet("Const"),
    "multidim": get_cell_dict_for_sheet("Multidim"),
}
