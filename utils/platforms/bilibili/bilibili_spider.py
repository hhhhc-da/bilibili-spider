from fake_useragent import UserAgent
import pandas as pd
import os, time, requests

# 解析 responses['data'], 测试案例如下
# {'cursor': {'is_begin': False, 'prev': 1, 'next': 2, 'is_end': True, 'pagination_reply': {'prev_offset': '{"type":1,"direction":2,"data":{"pn":1}}'}, 'session_id': '', 'mode': 3, 'mode_text': '', 'all_count': 10, 'support_mode': [2, 3], 'name': '热门评论'}, 'replies': [{'rpid': 241633169041, 'oid': 112801259522492, 'type': 1, 'mid': 502311274, 'root': 0, 'parent': 0, 'dialog': 0, 'count': 1, 'rcount': 1, 'state': 0, 'fansgrade': 0, 'attr': 512, 'ctime': 1725768075, 'mid_str': '502311274', 'oid_str': '112801259522492', 'rpid_str': '241633169041', 'root_str': '0', 'parent_str': '0', 'dialog_str': '0', 'like': 0, 'action': 0, 'member': {'mid': '502311274', 'uname': 'Thunderstrike', 'sex': '保密', 'sign': '个人简历：00年出生02会实变函数03年获得麻省理工博士学位 04年获得特异功能 05年得道成仙..23年还想做一次这样的白日梦', 'avatar': 'https://i1.hdslb.com/bfs/face/15198778f02dd75d6fa5c37f3b0ade9b60d49b7f.jpg', 'rank': '10000', 'face_nft_new': 0, 'is_senior_member': 0, 'senior': {}, 'level_info': {'current_level': 5, 'current_min': 0, 'current_exp': 0, 'next_exp': 0}, 'pendant': {'pid': 0, 'name': '', 'image': '', 'expire': 0, 'image_enhance': '', 'image_enhance_frame': '', 'n_pid': 0}, 'nameplate': {'nid': 0, 'name': '', 'image': '', 'image_small': '', 'level': '', 'condition': ''}, 'official_verify': {'type': -1, 'desc': ''}, 'vip': {'vipType': 1, 'vipDueDate': 1707667200000, 'dueRemark': '', 'accessStatus': 0, 'vipStatus': 0, 'vipStatusWarn': '', 'themeType': 0, 'label': {'path': '', 'text': '', 'label_theme': '', 'text_color': '', 'bg_style': 0, 'bg_color': '', 'border_color': '', 'use_img_label': True, 'img_label_uri_hans': '', 'img_label_uri_hant': '', 'img_label_uri_hans_static': 'https://i0.hdslb.com/bfs/vip/d7b702ef65a976b20ed854cbd04cb9e27341bb79.png', 'img_label_uri_hant_static': 'https://i0.hdslb.com/bfs/activity-plat/static/20220614/e369244d0b14644f5e1a06431e22a4d5/KJunwh19T5.png'}, 'avatar_subscript': 0, 'nickname_color': ''}, 'fans_detail': None, 'user_sailing': {'pendant': None, 'cardbg': None, 'cardbg_with_focus': None}, 'user_sailing_v2': {}, 'is_contractor': False, 'contract_desc': '', 'nft_interaction': None, 'avatar_item': {'container_size': {'width': 1.8, 'height': 1.8}, 'fallback_layers': {'layers': [{'visible': True, 'general_spec': {'pos_spec': {'coordinate_pos': 2, 'axis_x': 0.9, 'axis_y': 0.9}, 'size_spec': {'width': 1, 'height': 1}, 'render_spec': {'opacity': 1}}, 'layer_config': {'tags': {'AVATAR_LAYER': {}, 'GENERAL_CFG': {'config_type': 1, 'general_config': {'web_css_style': {'borderRadius': '50%'}}}}, 'is_critical': True}, 'resource': {'res_type': 3, 'res_image': {'image_src': {'src_type': 1, 'placeholder': 6, 'remote': {'url': 'https://i1.hdslb.com/bfs/face/15198778f02dd75d6fa5c37f3b0ade9b60d49b7f.jpg', 'bfs_style': 'widget-layer-avatar'}}}}}], 'is_critical_group': True}, 'mid': '502311274'}}, 'content': {'message': '是不是用python直接apt 就可以[doge_金箍][黑眼圈_金箍]', 'members': [], 'emote': {'[doge_金箍]': {'id': 83964, 'package_id': 1, 'state': 0, 'type': 1, 'attr': 0, 'text': '[doge_金箍]', 'url': 'https://i0.hdslb.com/bfs/emote/aadaca1895e09c5596fc6365192ec93a23718cf0.png', 'meta': {'size': 1, 'suggest': ['']}, 'mtime': 1724910828, 'jump_title': '金箍'}, '[黑眼圈_金箍]': {'id': 83965, 'package_id': 1, 'state': 0, 'type': 1, 'attr': 0, 'text': '[黑眼圈_金箍]', 'url': 'https://i0.hdslb.com/bfs/emote/3d8edacc6efa4bc397642ee2bdc688c2eb976b4f.png', 'meta': {'size': 1, 'suggest': ['']}, 'mtime': 1724918325, 'jump_title': '金箍'}}, 'jump_url': {'apt': {'title': 'apt', 'state': 0, 'prefix_icon': 'https://i0.hdslb.com/bfs/reply/9f3ad0659e84c96a711b88dd33f4bc2e945045e0.png', 'app_url_schema': 'bilibili://search?from=appcommentline_search&search_from_source=appcommentline_search&direct_return=true&keyword=apt&seid=4794086247030911852', 'app_name': '', 'app_package_name': '', 'click_report': '', 'is_half_screen': False, 'exposure_report': '', 'extra': {'goods_show_type': 0, 'is_word_search': True, 'goods_cm_control': 0, 'goods_click_report': '', 'goods_exposure_report': ''}, 'underline': False, 'match_once': True, 'pc_url': '//search.bilibili.com/all?from_source=webcommentline_search&keyword=apt&seid=4794086247030911852', 'icon_position': 1}, 'python': {'title': 'python', 'state': 0, 'prefix_icon': 'https://i0.hdslb.com/bfs/reply/9f3ad0659e84c96a711b88dd33f4bc2e945045e0.png', 'app_url_schema': 'bilibili://search?from=appcommentline_search&search_from_source=appcommentline_search&direct_return=true&keyword=python&seid=4794086247030911852', 'app_name': '', 'app_package_name': '', 'click_report': '', 'is_half_screen': False, 'exposure_report': '', 'extra': {'goods_show_type': 0, 'is_word_search': True, 'goods_cm_control': 0, 'goods_click_report': '', 'goods_exposure_report': ''}, 'underline': False, 'match_once': True, 'pc_url': '//search.bilibili.com/all?from_source=webcommentline_search&keyword=python&seid=4794086247030911852', 'icon_position': 1}}, 'max_line': 6}, 'replies': [{'rpid': 242830090433, 'oid': 112801259522492, 'type': 1, 'mid': 397213296, 'root': 241633169041, 'parent': 241633169041, 'dialog': 242830090433, 'count': 0, 'rcount': 0, 'state': 0, 'fansgrade': 0, 'attr': 0, 'ctime': 1727079240, 'mid_str': '397213296', 'oid_str': '112801259522492', 'rpid_str': '242830090433', 'root_str': '241633169041', 'parent_str': '241633169041', 'dialog_str': '242830090433', 'like': 0, 'action': 0, 'member': {'mid': '397213296', 'uname': '不能吃的笨蛋菜乃花', 'sex': '保密', 'sign': '', 'avatar': 'https://i0.hdslb.com/bfs/face/377311532bc568ee08630e48d3df042bb6cbf6b3.jpg', 'rank': '10000', 'face_nft_new': 0, 'is_senior_member': 0, 'senior': {}, 'level_info': {'current_level': 5, 'current_min': 0, 'current_exp': 0, 'next_exp': 0}, 'pendant': {'pid': 0, 'name': '', 'image': '', 'expire': 0, 'image_enhance': '', 'image_enhance_frame': '', 'n_pid': 0}, 'nameplate': {'nid': 0, 'name': '', 'image': '', 'image_small': '', 'level': '', 'condition': ''}, 'official_verify': {'type': -1, 'desc': ''}, 'vip': {'vipType': 0, 'vipDueDate': 0, 'dueRemark': '', 'accessStatus': 0, 'vipStatus': 0, 'vipStatusWarn': '', 'themeType': 0, 'label': {'path': '', 'text': '', 'label_theme': '', 'text_color': '', 'bg_style': 0, 'bg_color': '', 'border_color': '', 'use_img_label': True, 'img_label_uri_hans': '', 'img_label_uri_hant': '', 'img_label_uri_hans_static': 'https://i0.hdslb.com/bfs/vip/d7b702ef65a976b20ed854cbd04cb9e27341bb79.png', 'img_label_uri_hant_static': 'https://i0.hdslb.com/bfs/activity-plat/static/20220614/e369244d0b14644f5e1a06431e22a4d5/KJunwh19T5.png'}, 'avatar_subscript': 0, 'nickname_color': ''}, 'fans_detail': None, 'user_sailing': None, 'is_contractor': False, 'contract_desc': '', 'nft_interaction': None, 'avatar_item': {'container_size': {'width': 1.8, 'height': 1.8}, 'fallback_layers': {'layers': [{'visible': True, 'general_spec': {'pos_spec': {'coordinate_pos': 2, 'axis_x': 0.9, 'axis_y': 0.9}, 'size_spec': {'width': 1, 'height': 1}, 'render_spec': {'opacity': 1}}, 'layer_config': {'tags': {'AVATAR_LAYER': {}, 'GENERAL_CFG': {'config_type': 1, 'general_config': {'web_css_style': {'borderRadius': '50%'}}}}, 'is_critical': True}, 'resource': {'res_type': 3, 'res_image': {'image_src': {'src_type': 1, 'placeholder': 6, 'remote': {'url': 'https://i0.hdslb.com/bfs/face/377311532bc568ee08630e48d3df042bb6cbf6b3.jpg', 'bfs_style': 'widget-layer-avatar'}}}}}], 'is_critical_group': True}, 'mid': '397213296'}}, 'content': {'message': 'python一键pip就行，但是有时候会遇到版本不对或者依赖编译不了的情况[doge]有一些时候会遇到', 'members': [], 'emote': {'[doge]': {'id': 26, 'package_id': 1, 'state': 0, 'type': 1, 'attr': 0, 'text': '[doge]', 'url': 'https://i0.hdslb.com/bfs/emote/3087d273a78ccaff4bb1e9972e2ba2a7583c9f11.png', 'meta': {'size': 1, 'suggest': ['']}, 'mtime': 1668688325, 'jump_title': 'doge'}}, 'jump_url': {}, 'max_line': 3}, 'replies': None, 'assist': 0, 'up_action': {'like': False, 'reply': False}, 'invisible': False, 'reply_control': {'max_line': 3, 'time_desc': '179天 前发布', 'location': 'IP属地：天津'}, 'folder': {'has_folded': False, 'is_folded': False, 'rule': ''}, 'dynamic_id_str': '0', 'note_cvid_str': '0', 'track_info': ''}], 'assist': 0, 'up_action': {'like': False, 'reply': True}, 'invisible': False, 'reply_control': {'up_reply': True, 'max_line': 6, 'sub_reply_entry_text': '共 1 条回复', 'sub_reply_title_text': '相关回复共1条', 'time_desc': '194天前发布', 'location': 'IP属地：广东'}, 'folder': {'has_folded': False, 'is_folded': False, 'rule': ''}, 'dynamic_id_str': '0', 'note_cvid_str': '0', 'track_info': ''}, {'rpid': 232943828752, 'oid': 112801259522492, 'type': 1, 'mid': 23242745, 'root': 0, 'parent': 0, 'dialog': 0, 'count': 2, 'rcount': 2, 'state': 0, 'fansgrade': 0, 'attr': 512, 'ctime': 1721632945, 'mid_str': '23242745', 'oid_str': '112801259522492', 'rpid_str': '232943828752', 'root_str': '0', 'parent_str': '0', 'dialog_str': '0', 'like': 0, 'action': 0, 'member': {'mid': '23242745', 'uname': '反应太快', 'sex': '男', 'sign': '我们必将知道，我们即将知道 。', 'avatar': 'https://i2.hdslb.com/bfs/face/9e71689fa38bc2a96da9c690fc744fbf0e7653b2.jpg', 'rank': '10000', 'face_nft_new': 0, 'is_senior_member': 1, 'senior': {'status': 2}, 'level_info': {'current_level': 6, 'current_min': 0, 'current_exp': 0, 'next_exp': 0}, 'pendant': {'pid': 2630, 'name': '三周年恋曲', 'image': 'https://i2.hdslb.com/bfs/garb/item/da59339168e3c369e121c0dfa5b7e70ab4bd6a2e.png', 'expire': 0, 'image_enhance': 'https
# {'cursor': {'is_begin': False, 'prev': 2, 'next': 3, 'is_end': True, 'pagination_reply': {'prev_offset': '{"type":1,"direction":2,"data":{"pn":2}}'}, 'session_id': '', 'mode': 3, 'mode_text': '', 'all_count': 10, 'support_mode': [2, 3], 'name': '热门评论'}, 'replies': [], 'top': {'admin': None, 'upper': None, 'vote': None}, 'top_replies': [], 'up_selection': {'pending_count': 0, 'ignore_count': 0}, 'effects': {'preloading': ''}, 'assist': 0, 'blacklist': 0, 'vote': 1, 'config': {'showtopic': 1, 'show_up_flag': True, 'read_only': False}, 'upper': {'mid': 397213296}, 'control': {'input_disable': False, 'root_input_text': '评论千万条，等你发一条', 'child_input_text': '评论千万条，等你发一条', 'giveup_input_text': '不发没关 系，请继续友善哦~', 'screenshot_icon_state': 1, 'upload_picture_icon_state': 1, 'answer_guide_text': '需要升级成为lv2会员后才可以评论，先去答题转 正吧！', 'answer_guide_icon_url': 'http://i0.hdslb.com/bfs/emote/96940d16602cacbbac796245b7bb99fa9b5c970c.png', 'answer_guide_ios_url': 'https://www.bilibili.com/h5/newbie/entry?navhide=1&re_src=12', 'answer_guide_android_url': 'https://www.bilibili.com/h5/newbie/entry?navhide=1&re_src=6', 'bg_text': '', 'empty_page': None, 'show_type': 1, 'show_text': '', 'web_selection': False, 'disable_jump_emote': False, 'enable_charged': False, 'enable_cm_biz_helper': False, 'preload_resources': None}, 'note': 1, 'esports_grade_card': None, 'callbacks': None, 'context_feature': ''}

def extract_comments_data(comments, response):
    # 遍历主评论
    for reply in response.get('replies', []):
        member = reply.get('member', {})
        content = reply.get('content', {})
        
        # 提取主评论信息
        comments.append({
            "username": member.get('uname', ''),
            "uid": member.get('mid', ''),
            "sex": member.get('sex', ''),
            "comment": content.get('message', '')
        })
        
        # 遍历子评论（如果有）
        sub_replies = reply.get('replies', []) or []
        for sub_reply in sub_replies:
            sub_member = sub_reply.get('member', {})
            sub_content = sub_reply.get('content', {})
            
            comments.append({
                "username": sub_member.get('uname', ''),
                "uid": sub_member.get('mid', ''),
                "sex": sub_member.get('sex', ''),
                "comment": sub_content.get('message', '')
            })
            
    return comments


def extract_comments(comment, replies):
    """
    递归提取评论内容，处理多级评论
    """
    for content in replies:
        # 添加当前评论
        comment.append(content["content"]["message"])
        # 如果有嵌套的子评论，递归提取
        if "replies" in content and content["replies"]:
            extract_comments(comment, content["replies"])


def spider_from_oid(bvid="BV1kQFUeaEeW", oid="112801259522492", base_dir=os.path.join("xlsx", "output"), file_name='bilibili_output.xlsx', cookie=""):
    '''
    爬虫主程序, 输入视频 OID 进行爬取, Cookie 需要替换为有效的, OID 的获取可以在控制台 NetWork 搜索 oid
    '''
    comment = []
    pre_comment_length = 0
    
    # 检查是否存在文件夹
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    # url 拼接, 使用 next 控制翻页
    url = "https://api.bilibili.com/x/v2/reply/main?csrf=40a227fcf12c380d7d3c81af2cd8c5e8&mode=3&next={}&oid=" + str(oid) + "&plat=1&type=1"

    header = {
        "user-agent": UserAgent().random,
        "cookie": cookie
    }
    
    # B站的第一页是 1
    i = 1

    while True:
        # 添加重试机制可以让爬虫收集完全部的一级评论，且不会中途停止
        try:
            responses = requests.get(url=url.format(i), headers=header).json()
        except:
            time.sleep(1)
            continue
        i += 1  # 获取下一页评论

        # 提取当前页面的评论（包括所有嵌套评论）
        if "data" in responses and "replies" in responses["data"]:
            comment = extract_comments_data(comment, responses["data"])
        
        # 判断是否已收集完所有评论
        if len(comment) == pre_comment_length:
            print(f"OID: {oid} 搜集到评论数: {len(comment)}")
            break
        else:
            pre_comment_length = len(comment)

        # 延迟爬虫速度，避免频繁请求被封禁
        time.sleep(1)

    # 保存所有评论到文件
    pf = pd.DataFrame(comment)

    with pd.ExcelWriter(os.path.join(base_dir, file_name), engine='openpyxl', mode="a" if os.path.exists(os.path.join(base_dir, file_name)) else "w") as writer:
        # 将 DataFrame 写入 Excel
        pf.to_excel(writer, sheet_name='{}'.format(bvid), index=False)
        
        # 获取工作簿和工作表对象
        workbook  = writer.book
        worksheet = workbook['{}'.format(bvid)]
        # 设置列宽
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 20
        worksheet.column_dimensions['D'].width = 100
        
    print(f"Excel 文件 (文件名: {file_name}) 已保存")