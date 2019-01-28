/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glow/Backends/QueueBackedDeviceManager.h"

using namespace glow;
using namespace glow::runtime;

template <class F> struct shared_function {
  std::shared_ptr<F> f;
  shared_function() = delete; // = default works, but I don't use it
  shared_function(F &&f_) : f(std::make_shared<F>(std::move(f_))) {}
  shared_function(shared_function const &) = default;
  shared_function(shared_function &&) = default;
  shared_function &operator=(shared_function const &) = default;
  shared_function &operator=(shared_function &&) = default;
  template <class... As> auto operator()(As &&... as) const {
    return (*f)(std::forward<As>(as)...);
  }
};
template <class F>
shared_function<std::decay_t<F>> make_shared_function(F &&f) {
  return {std::forward<F>(f)};
}
QueueBackedDeviceManager::QueueBackedDeviceManager(BackendKind backend,
                                                   llvm::StringRef name)
    : DeviceManager(backend, name), workThread_(1) {}

QueueBackedDeviceManager::~QueueBackedDeviceManager() {
  stop(true); // will join workThread_
}

void QueueBackedDeviceManager::init() {}

void QueueBackedDeviceManager::addNetwork(const Module *module,
                                          FunctionMapTy functions,
                                          ReadyCBTy callback) {
  workThread_.submit([this, module, f = std::move(functions),
                      c = std::move(callback)]() mutable {
    addNetworkImpl(module, std::move(f), std::move(c));
  });
}

void QueueBackedDeviceManager::evictNetwork(llvm::StringRef functionName) {
  workThread_.submit([this, functionName] { evictNetworkImpl(functionName); });
}

RunIdentifierTy
QueueBackedDeviceManager::runFunction(std::string functionName,
                                      std::unique_ptr<Context> ctx,
                                      ResultCBTy callback) {

  RunIdentifierTy id = nextIdentifier_++;
  auto func = [this, id, functionName = std::move(functionName),
               ctx = std::move(ctx), callback = std::move(callback)]() mutable {
    runFunctionImpl(id, std::move(functionName), std::move(ctx),
                    std::move(callback));
  };
  std::function<void()> funcShared = make_shared_function(std::move(func));
  workThread_.submit(funcShared);
  return id;
}

void QueueBackedDeviceManager::stop(bool block) { workThread_.stop(block); }
