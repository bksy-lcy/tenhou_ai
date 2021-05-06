yaku_dict_jp = {
    "Aka Dora": "赤ドラ",
    "Chankan": "槍槓",
    "Chiitoitsu": "七対子",
    "Chantai": "混全帯么九",
    "Chinitsu": "清一色",
    "Yakuhai (chun)": "役牌(中)",
    "Double Riichi": "ダブル立直",
    "Dora": "ドラ",
    "Yakuhai (east)": "役牌(東)",
    "Haitei Raoyue": "海底摸月",
    "Yakuhai (haku)": "役牌(白)",
    "Yakuhai (hatsu)": "役牌(發)",
    "Honitsu": "混一色",
    "Honroutou": "混老頭",
    "Houtei Raoyui": "河底撈魚",
    "Iipeiko": "一盃口",
    "Ippatsu": "一発",
    "Ittsu": "一気通貫",
    "Junchan": "純全帯么九",
    "Nagashi Mangan": "流し満貫",
    "Yakuhai (north)": "役牌(北)",
    "Pinfu": "平和",
    "Renhou": "人和",
    "Riichi": "立直",
    "Rinshan Kaihou": "嶺上開花",
    "Ryanpeikou": "二盃口",
    "San Ankou": "三暗刻",
    "San Kantsu": "三槓子",
    "Sanshoku Doujun": "三色同順",
    "Sanshoku Doukou": "三色同刻",
    "Shou Sangen": "小三元",
    "Yakuhai (south)": "役牌(南)",
    "Tanyao": "断么九",
    "Toitoi": "対々和",
    "Menzen Tsumo": "門前清自摸和",
    "Yakuhai (west)": "役牌(西)",
    "Yakuhai (wind of place)": "自風",
    "Yakuhai (wind of round)": "場風",
    "Chiihou": "地和",
    "Chinroutou": "清老頭",
    "Chuuren Poutou": "九蓮宝燈",
    "Daburu Chuuren Poutou": "純正九蓮宝燈",
    "Kokushi Musou Juusanmen Matchi": "国士無双十三面待ち",
    "Daisangen": "大三元",
    "Daisharin": "大車輪",
    "Daisuurin": "大数隣",
    "Daichikurin": "大竹林",
    "Dai Suushii": "大四喜",
    "Kokushi Musou": "国士無双",
    "Renhou (yakuman)": "人和(役満)",
    "Ryuuiisou": "緑一色",
    "Shousuushii": "小四喜",
    "Suu Ankou": "四暗刻",
    "Suu Ankou Tanki": "四暗刻単騎待ち",
    "Suu Kantsu": "四槓子",
    "Tenhou": "天和",
    "Tsuu Iisou": "字一色",
}

fu_dict_jp = {
    "base": "基本符",
    "penchan": "ペンチャン",
    "kanchan": "カンチャン",
    "valued_pair": "役牌雀頭",
    "double_valued_pair": "連風雀頭",
    "pair_wait": "単騎",
    "tsumo": "ツモ",
    "hand_without_fu": "門前なく平和ようなロン",
    "closed_pon": "チュンチャン暗刻",
    "open_pon": "チュンチャン明刻",
    "closed_terminal_pon": "么九暗刻",
    "open_terminal_pon": "么九明刻",
    "closed_kan": "チュンチャン暗槓",
    "open_kan": "チュンチャン明槓",
    "closed_terminal_kan": "么九暗槓",
    "open_terminal_kan": "么九明槓",
}

cost_dict_jp = {
    "fu": "符",
    "han": "飜",
    "point": "点",
    "dealer_pays": "親の支払い",
    "player_pays": "子の支払い",
    "trigger_pays": "銃の支払い",
    "kyoutaku_bonus": "キョウタク",
    "total": "総計",
    "fu_details": "【符詳しく】",
    "cost_details": "【得点詳しく】",
    "yaku_details": "【役詳しく】",
    "unavailable": "不可用",
    "kazoe yakuman": "累计役满",
    "kazoe sanbaiman": "累计三倍满",
    "6x yakuman": "6倍役満",
    "5x yakuman": "5倍役満",
    "4x yakuman": "4倍役満",
    "3x yakuman": "3倍役満",
    "2x yakuman": "2倍役満",
    "yakuman": "役満",
    "sanbaiman": "三倍満",
    "baiman": "倍満",
    "haneman": "跳満",
    "mangan": "満貫",
    "kiriage mangan": "切り上げ満貫",
    "nagashi mangan": "流し満貫",
}

err_dict_jp = {
    "Error": "誤り",
    "winning_tile_not_in_hand": "待ち牌が手にありません",
    "open_hand_riichi_not_allowed": "副露した状態はリーチできません",
    "open_hand_daburi_not_allowed": "副露した状態はダブルリーチできません",
    "ippatsu_without_riichi_not_allowed": "一発はリーチの指定が必要です",
    "hand_not_winning": "手があがりません",
    "no_yaku": "役がありません",
    "chankan_with_tsumo_not_allowed": "槍槓とツモは同時に指定できません",
    "rinshan_without_tsumo_not_allowed": "嶺上開花にはツモの指定が必要です",
    "haitei_without_tsumo_not_allowed": "海底摸月にはツモの指定が必要です",
    "houtei_with_tsumo_not_allowed": "河底撈魚とツモは同時に指定できません",
    # A dead wall tile is not considered as "the last tile"
    "haitei_with_rinshan_not_allowed": "海底摸月と嶺上開花は同時に指定できません",
    # You can't make a meld in the last turn
    "houtei_with_chankan_not_allowed": "河底撈魚と槍槓は同時に指定できません",
    "tenhou_not_as_dealer_not_allowed": "子は天和を指定できません",
    "tenhou_without_tsumo_not_allowed": "天和にはツモの指定が必要です",
    "tenhou_with_meld_not_allowed": "天和は副露した状態で指定できません",
    "chiihou_as_dealer_not_allowed": "親は地和を指定できません",
    "chiihou_without_tsumo_not_allowed": "地和にはツモの指定が必要です",
    "chiihou_with_meld_not_allowed": "地和は副露した状態で指定できません",
    "renhou_as_dealer_not_allowed": "親は人和を指定できません",
    "renhou_with_tsumo_not_allowed": "人和とツモは同時に指定できません",
    "renhou_with_meld_not_allowed": "人和は副露した状態で指定できません",
}
