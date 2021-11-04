import pyvideoai.models.timecycle.videos.model_test as video3d

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def main():
    model = video3d.CycleTime(trans_param_num=3)
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    title = 'videonet'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        partial_load(checkpoint['state_dict'], model)

        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])

        del checkpoint

    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Contrast Loss'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss = test(val_loader, model, 1, use_cuda)


def img_unnormalise(img, mean, std):
    """
    Unnormalise images for visualisation.
    """
    final_img = img.copy()
    
    for c in range(3):
        final_img[c] = final_img[c] * std[c]
        final_img[c] = final_img[c] + mean[c]

    final_img = final_img * 255
    final_img = np.transpose(final_img, (1, 2, 0))
    final_img = cv2.resize(final_img, (final_img.shape[0] * 2, final_img.shape[1] * 2) )
    return final_img


def vis_coordinate_source(x, y, factor=8):
    return x*factor + factor//2, y*factor + factor//2

def vis_coordinate_target(x, y, factor=8, img_width=200, border_width=10):
    x, y = vis_coordinate_source(x, y, factor)
    return x + border_width + img_width, y

def test(val_loader, model, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    save_objs = args.evaluate

    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)# /scratch/xiaolonw/davis_results_mask_mixfcn/')
    # save_path = '/scratch/xiaolonw/davis_results_mask_mixfcn/'
    save_path = args.save_path + '/'
    # img_path  = '/scratch/xiaolonw/vlog_frames/'
    save_file = '%s/list.txt' % save_path

    fileout = open(save_file, 'w')

    end = time.time()

    # bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (imgs_total, patch2_total, lbls, meta) in enumerate(val_loader):

        finput_num_ori = params['videoLen']
        finput_num     = finput_num_ori

        # measure data loading time
        data_time.update(time.time() - end)
        imgs_total = torch.autograd.Variable(imgs_total.cuda())
        # patch2_total = torch.autograd.Variable(patch2_total.cuda())

        t00 = time.time()

        bs = imgs_total.size(0)
        total_frame_num = imgs_total.size(1)
        channel_num = imgs_total.size(2)
        height_len  = imgs_total.size(3)
        width_len   = imgs_total.size(4)

        assert(bs == 1)

        folder_paths = meta['folder_path']
        gridx = int(meta['gridx'].data.cpu().numpy()[0])
        gridy = int(meta['gridy'].data.cpu().numpy()[0])
        print('gridx: ' + str(gridx) + ' gridy: ' + str(gridy))
        print('total_frame_num: ' + str(total_frame_num))

        height_dim = int(params['cropSize'] / 8)
        width_dim  = int(params['cropSize'] / 8)

        t02 = time.time()

        # print the images

        imgs_set = imgs_total.data
        imgs_set = imgs_set.cpu().numpy()
        imgs_set = imgs_set[0]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        imgs_toprint = []

        # ref image
        for t in range(imgs_set.shape[0]):
            img_now = imgs_set[t]

            for c in range(3):
                img_now[c] = img_now[c] * std[c]
                img_now[c] = img_now[c] + mean[c]

            img_now = img_now * 255
            img_now = np.transpose(img_now, (1, 2, 0))
            img_now = cv2.resize(img_now, (img_now.shape[0] * 2, img_now.shape[1] * 2) )

            imgs_toprint.append(img_now)

            imname  = save_path + str(batch_idx) + '_' + str(t) + '_frame.jpg'
            scipy.misc.imsave(imname, img_now)

        now_batch_size = 4

        imgs_stack = []
        patch2_stack = []

        im_num = total_frame_num - finput_num_ori

        trans_out_2_set = []
        corrfeat2_set = []

        imgs_tensor = torch.Tensor(now_batch_size, 1, 3, params['cropSize'], params['cropSize'])
        target_tensor = torch.Tensor(now_batch_size, 1, 3, params['cropSize'], params['cropSize'])

        imgs_tensor = torch.autograd.Variable(imgs_tensor.cuda())
        target_tensor = torch.autograd.Variable(target_tensor.cuda())


        t03 = time.time()

        for iter in range(0, im_num, now_batch_size):

            print(iter)

            startid = iter
            endid   = iter + now_batch_size

            if endid > im_num:
                endid = im_num

            now_batch_size2 = endid - startid

            for i in range(now_batch_size2):

                imgs = imgs_total[:, iter + i + finput_num_ori-1, :, :, :].unsqueeze(1)
                imgs_tensor[i] = imgs
                target_tensor[i, 0] = imgs_total[0, iter + i + finput_num_ori]

            corrfeat2_now = model(imgs_tensor, target_tensor)
            corrfeat2_now = corrfeat2_now.view(now_batch_size, corrfeat2_now.size(1), corrfeat2_now.size(2), corrfeat2_now.size(3))
            #print(corrfeat2_now.shape)

            for i in range(now_batch_size2):
                corrfeat2_set.append(corrfeat2_now[i].data.cpu().numpy())

                # visualise top1
                corr = corrfeat2_set[-1]
                _, corr_size_H, corr_size_W = corr.shape
                assert corr_size_H == corr_size_W
                corr_size = corr_size_H
                corr = corr.reshape(corr_size**2, corr_size**2)
                corr_2d = corr.reshape(corr_size_W, corr_size_H, corr_size_H, corr_size_W)

                img = imgs_tensor[i, 0]         # (C, H, W)
                target = target_tensor[i, 0]    # (C, H, W)

                img = img_unnormalise(img.cpu().numpy(), mean, std)
                target = img_unnormalise(target.cpu().numpy(), mean, std)

                H, W, C = img.shape

                vis_corr = np.zeros((H, W * 2 + 10, C))
                vis_corr[:, :W, :] = img
                vis_corr[:, -W:, :] = target 
                vis_corr = np.ascontiguousarray(vis_corr[...,::-1])       # BGR

                vis_arrowflow = np.copy(vis_corr)
                vis_filteredarrowflow = np.copy(vis_corr)
                vis_attentionmap = np.copy(vis_corr)

                topk_target = 1
                ids = np.argpartition(corr, -topk_target, axis=1)[:, -topk_target:]
                corr_top1 = np.take_along_axis(corr, ids, axis=1).flatten()

                # for correspondense match visualisation. Filter topk_source points.
                topk_source = 15
                srcids = np.argpartition(corr_top1, -topk_source)[-topk_source:]

                colours = colormap.colormap(rgb=False, maximum=255)

                for source_idx, target_idx in enumerate(ids):
                    colour = tuple(np.round(colours[source_idx % colours.shape[0]]).astype('int'))
                    for n in range(topk_target):
                        source_x = source_idx // corr_size
                        source_y = source_idx % corr_size
                        target_y = target_idx // corr_size
                        target_x = target_idx % corr_size
                        #print(source_y, source_x, target_y[n], target_x[n])

                        vis_x_src, vis_y_src = vis_coordinate_source(source_x, source_y, factor=8*2)
                        vis_x_tgt, vis_y_tgt = vis_coordinate_target(target_x[n], target_y[n], factor=8*2, img_width=200*2)

                        if source_idx in srcids:
                            vis_corr = cv2.circle(vis_corr, (vis_x_src, vis_y_src), 3, colour, thickness=1)
                            vis_corr = cv2.circle(vis_corr, (vis_x_tgt, vis_y_tgt), 3, colour, thickness=1)
                            vis_corr = cv2.line(vis_corr, (vis_x_src, vis_y_src), (vis_x_tgt, vis_y_tgt), colour, thickness=1)

                        vis_x_tgt, vis_y_tgt = vis_coordinate_source(target_x[n], target_y[n], factor=8*2)
                        vis_arrowflow = cv2.arrowedLine(vis_arrowflow, (vis_x_src, vis_y_src), (vis_x_tgt, vis_y_tgt), colour, thickness=1)
                        if corr_2d[source_y, source_x, target_x, target_y] > 0.5:
                            vis_filteredarrowflow = cv2.arrowedLine(vis_arrowflow, (vis_x_src, vis_y_src), (vis_x_tgt, vis_y_tgt), colour, thickness=1)

                imname  = save_path + str(batch_idx) + '_' + str(iter + i + finput_num_ori) + '_corr.jpg'
                cv2.imwrite(imname, vis_corr)
                imname  = save_path + str(batch_idx) + '_' + str(iter + i + finput_num_ori) + '_arrowflow.jpg'
                cv2.imwrite(imname, vis_arrowflow)
                imname  = save_path + str(batch_idx) + '_' + str(iter + i + finput_num_ori) + '_filteredarrowflow.jpg'
                cv2.imwrite(imname, vis_filteredarrowflow)

                #attentionmap = np.sum(corr, axis=1).reshape(corr_size_W, corr_size_H).transpose(1,0)
                #print(np.max(attentionmap))        # around 3
                #attentionmap = attentionmap / 3
                attentionmap = corr_top1.reshape(corr_size_W, corr_size_H).transpose(1,0).astype('float64')
                attentionmap = cv2.resize(attentionmap, (W, H))
                #attentionmap = attentionmap 
                #for c in range(3):
                vis_attentionmap[:H, :W, :] *= attentionmap[:,:,None]


                imname  = save_path + str(batch_idx) + '_' + str(iter + i + finput_num_ori) + '_attentionmap.jpg'
                cv2.imwrite(imname, vis_attentionmap)






        t04 = time.time()
        print(t04-t03, 'model forward', t03-t02, 'image prep')


        if False:
            for iter in range(total_frame_num - finput_num_ori):

                if iter % 10 == 0:
                    print(iter)

                imgs = imgs_total[:, iter + 1: iter + finput_num_ori, :, :, :]
                imgs2 = imgs_total[:, 0, :, :, :].unsqueeze(1)
                imgs = torch.cat((imgs2, imgs), dim=1)

                # trans_out_2, corrfeat2 = model(imgs, patch2)
                corrfeat2   = corrfeat2_set[iter]
                corrfeat2   = torch.from_numpy(corrfeat2)


                out_frame_num = int(finput_num)
                height_dim = corrfeat2.size(2)
                width_dim = corrfeat2.size(3)

                corrfeat2 = corrfeat2.view(corrfeat2.size(0), height_dim, width_dim, height_dim, width_dim)
                corrfeat2 = corrfeat2.data.cpu().numpy()


                topk_vis = args.topk_vis
                vis_ids_h = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)
                vis_ids_w = np.zeros((corrfeat2.shape[0], height_dim, width_dim, topk_vis)).astype(np.int)

                t05 = time.time()

                atten1d  = corrfeat2.reshape(corrfeat2.shape[0], height_dim * width_dim, height_dim, width_dim)
                print(atten1d.shape)
                ids = np.argpartition(atten1d, -topk_vis, axis=1)[:, -topk_vis:]
                print(ids.shape)
                # ids = np.argsort(atten1d, axis=1)[:, -topk_vis:]

                hid = ids // width_dim
                wid = ids % width_dim

                vis_ids_h = wid.transpose(0, 2, 3, 1)
                vis_ids_w = hid.transpose(0, 2, 3, 1)

                t06 = time.time()

                img_now = imgs_toprint[iter + finput_num_ori]

                predlbls = np.zeros((height_dim, width_dim, len(lbl_set)))
                # predlbls2 = np.zeros((height_dim * width_dim, len(lbl_set)))

                for t in range(finput_num):

                    tt1 = time.time()

                    h, w, k = np.meshgrid(np.arange(height_dim), np.arange(width_dim), np.arange(topk_vis), indexing='ij')
                    h, w = h.flatten(), w.flatten()

                    hh, ww = vis_ids_h[t].flatten(), vis_ids_w[t].flatten()

                    if t == 0:
                        lbl = lbls_resize2[0, hh, ww, :]
                    else:
                        lbl = lbls_resize2[t + iter, hh, ww, :]

                    np.add.at(predlbls, (h, w), lbl * corrfeat2[t, ww, hh, h, w][:, None])

                t07 = time.time()
                # print(t07-t06, 'lbl proc', t06-t05, 'argsorts')

                predlbls = predlbls / finput_num

                for t in range(len(lbl_set)):
                    nowt = t
                    predlbls[:, :, nowt] = predlbls[:, :, nowt] - predlbls[:, :, nowt].min()
                    predlbls[:, :, nowt] = predlbls[:, :, nowt] / predlbls[:, :, nowt].max()


                lbls_resize2[iter + finput_num_ori] = predlbls

                predlbls_cp = predlbls.copy()
                predlbls_cp = cv2.resize(predlbls_cp, (params['imgSize'], params['imgSize']))
                predlbls_val = np.zeros((params['imgSize'], params['imgSize'], 3))

                ids = np.argmax(predlbls_cp[:, :, 1 : len(lbl_set)], 2)

                predlbls_val = np.array(lbl_set)[np.argmax(predlbls_cp, axis=-1)]
                predlbls_val = predlbls_val.astype(np.uint8)
                predlbls_val2 = cv2.resize(predlbls_val, (img_now.shape[0], img_now.shape[1]), interpolation=cv2.INTER_NEAREST)

                # activation_heatmap = cv2.applyColorMap(predlbls, cv2.COLORMAP_JET)
                img_with_heatmap =  np.float32(img_now) * 0.5 + np.float32(predlbls_val2) * 0.5

                imname  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_label.jpg'
                imname2  = save_path + str(batch_idx) + '_' + str(iter + finput_num_ori) + '_mask.png'

                scipy.misc.imsave(imname, np.uint8(img_with_heatmap))
                scipy.misc.imsave(imname2, np.uint8(predlbls_val))



    fileout.close()

    return losses.avg


if __name__ == '__main__':
    main()
