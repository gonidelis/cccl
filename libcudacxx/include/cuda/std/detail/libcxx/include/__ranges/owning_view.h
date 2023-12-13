// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_OWNING_VIEW_H
#define _LIBCUDACXX___RANGES_OWNING_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/movable.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/data.h"
#include "../__ranges/empty.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/size.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/move.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER > 2014

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _CCCL_STD_VER > 2017
template<range _Rp>
  requires movable<_Rp> && (!__is_std_initializer_list<remove_cvref_t<_Rp>>)
#else
  template<class _Rp, enable_if_t<range<_Rp>, int> = 0,
                      enable_if_t<movable<_Rp>, int> = 0,
                      enable_if_t<!__is_std_initializer_list<remove_cvref_t<_Rp>>, int> = 0>
#endif
class owning_view : public view_interface<owning_view<_Rp>> {
  _Rp __r_ = _Rp();

public:
#if _CCCL_STD_VER > 2017
  owning_view() requires default_initializable<_Rp> = default;
#else
  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(default_initializable<_Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr owning_view()
    noexcept(is_nothrow_default_constructible_v<_Range>)
      : view_interface<owning_view<_Rp>>() {}
#endif
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr owning_view(_Rp&& __r)
    noexcept(is_nothrow_move_constructible_v<_Rp>)
    : view_interface<owning_view<_Rp>>(), __r_(_CUDA_VSTD::move(__r)) {}

  owning_view(owning_view&&) = default;
  owning_view& operator=(owning_view&&) = default;

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Rp& base() & noexcept { return __r_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr const _Rp& base() const& noexcept { return __r_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Rp&& base() && noexcept { return _CUDA_VSTD::move(__r_); }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr const _Rp&& base() const&& noexcept { return _CUDA_VSTD::move(__r_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr iterator_t<_Rp> begin() { return _CUDA_VRANGES::begin(__r_); }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr sentinel_t<_Rp> end() { return _CUDA_VRANGES::end(__r_); }

  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(range<const _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() const { return _CUDA_VRANGES::begin(__r_); }
  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(range<const _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() const { return _CUDA_VRANGES::end(__r_); }

  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(invocable<_CUDA_VRANGES::__empty::__fn, _Range&>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty()
    { return _CUDA_VRANGES::empty(__r_); }
  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(invocable<_CUDA_VRANGES::__empty::__fn, const _Range&>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty() const
    { return _CUDA_VRANGES::empty(__r_); }

  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(sized_range<_Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size()
    { return _CUDA_VRANGES::size(__r_); }
  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(sized_range<const _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto size() const
    { return _CUDA_VRANGES::size(__r_); }

  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(contiguous_range<_Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto data()
    { return _CUDA_VRANGES::data(__r_); }
  _LIBCUDACXX_TEMPLATE(class _Range = _Rp)
    _LIBCUDACXX_REQUIRES(contiguous_range<const _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto data() const
    { return _CUDA_VRANGES::data(__r_); }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(owning_view);

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template<class _Rp>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<owning_view<_Rp>> = enable_borrowed_range<_Rp>;

#endif // _CCCL_STD_VER > 2014

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_OWNING_VIEW_H
