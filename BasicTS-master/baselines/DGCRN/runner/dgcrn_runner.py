import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner


class DGCRNRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for DCRNN: add setup_graph and teacher forcing."""

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            dict: keys that must be included: inputs, prediction, target
        """

        data = self.preprocessing(data)
        dist = data['dist']
        sigma = data['sigma']
        adj = data['adj']

        # preprocess
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape
        dist = self.to_running_device(dist)
        sigma = self.to_running_device(sigma)
        adj = self.to_running_device(adj)


        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # customized curriculum learning
        task_level = self.curriculum_learning(epoch)
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num,
                                  epoch=epoch, train=train, task_level=task_level, dist=dist, sigma=sigma, adj_sc=adj)
        # parse model return
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return}
        model_return["inputs"] = self.select_target_features(history_data)
        model_return["target"] = self.select_target_features(future_data)
        assert list(model_return["prediction"].shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        model_return = self.postprocessing(model_return)

        alpha = self.model.kernel_layer.alpha
        multy_msk = 0
        if hasattr(self.model.kernel_layer, "alpha3"):
            multy_msk = 1
            alpha = torch.stack(
                [self.model.kernel_layer.alpha, self.model.kernel_layer.alpha2, self.model.kernel_layer.alpha3], dim=0)
            logits = self.model.kernel_layer.logits
        else:
            alpha = self.model.kernel_layer.alpha
            logits = None

        return model_return, multy_msk, self.model.kernel_layer.use_alpha, alpha, logits
